import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict, Optional
from src.methods.iql_agent import IQLAgent
from blendrl.agents.blender_agent import BlenderActorCritic
from src.methods.registry import register_agent

@register_agent("blendrl_iql")
class BlendRLIQLAgent(IQLAgent):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        
        # Handle nested agent config for algorithm
        algorithm = self.get_cfg("algorithm", self.get_cfg("name", cfg.env.name))

        from blendrl.env_vectorized import VectorizedNudgeBaseEnv
        self.env = VectorizedNudgeBaseEnv.from_name(
            cfg.env.name, 
            n_envs=1, 
            mode=algorithm, 
            seed=cfg.seed
        )
        
        dummy_logic, dummy_neural = self.env.reset()
        self.observation_space = dummy_neural.shape[1:]
        self.logic_observation_space = dummy_logic.shape[1:]
        self.n_actions = self.env.n_actions if not callable(self.env.n_actions) else self.env.n_actions()
        
        self.model = BlenderActorCritic(
            self.env,
            cfg.env.rules,
            cfg.agent.actor_mode,
            cfg.agent.blender_mode,
            cfg.agent.blend_function,
            cfg.env.reasoner,
            self.device,
            architecture=cfg.env.architecture,
            cfg=cfg.agent
        )

        # Q and Value networks are still neural-only in their standard implementation
        # but we could potentially make them hybrid too. For now, keep them neural.
        # However, the actor is the hybrid part.

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        return self.model.get_action_and_value(obs, logic_obs, action)

    def get_value(self, obs, logic_obs=None):
        # We don't have a hybrid value network in the model, but we can compute it
        return self.value_network(obs)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        
        # Check if we need to self-organize CEW modules
        epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
        if self.current_epoch % epochs_per_interval == 0:
            datamodule = self.trainer.datamodule
            # Use a reasonably large sample for organization
            sample_size = min(len(datamodule.reader), 10000)
            if sample_size > 0:
                batch = datamodule.reader.sample(sample_size)
                # Use logic_obs if available, otherwise obs
                organize_obs = batch["logic_obs"] if batch["logic_obs"] is not None else batch["obs"]
                self.model.self_organize_cew_modules(organize_obs)
                
                # Re-initialize actor optimizer because CEW parameters changed (new FLCs)
                lr = self.get_cfg("lr", 3e-4)
                opt_q, opt_v, opt_a = self.optimizers()
                actor_params = list(self.model.policy_modules.parameters()) + \
                               list(self.model.blender.parameters())
                # Replace the internal state of opt_a with a new optimizer for the new parameters
                new_opt_a = optim.Adam(actor_params, lr=lr)
                # This is a bit hacky but Pytorch Lightning Manual Optimization allows it
                self.trainer.strategy.optimizers[2] = new_opt_a

    def training_step(self, batch, batch_idx):
        datamodule = self.trainer.datamodule
        cfg = self.cfg
        
        batch_size = self.get_cfg("batch_size", 256)
        real_batch = datamodule.reader.sample(batch_size)
        obs = real_batch["obs"]
        logic_obs = real_batch["logic_obs"]
        actions = real_batch["action"]
        rewards = real_batch["reward"]
        next_obs = real_batch["next_obs"]
        next_logic_obs = real_batch["next_logic_obs"]
        dones = real_batch["done"]
        
        opt_q, opt_v, opt_a = self.optimizers()
        
        # 1. Update Q-networks (Neural-only for now)
        with torch.no_grad():
            next_v = self.value_network(next_obs).view(-1)
            q_target = rewards + cfg.env.gamma * next_v * (1 - dones)
            
        current_q1 = self.q_network(obs)
        current_q2 = self.q_network2(obs)
        current_q1_a = current_q1.gather(1, actions.unsqueeze(1)).view(-1)
        current_q2_a = current_q2.gather(1, actions.unsqueeze(1)).view(-1)
        
        q_loss = F.mse_loss(current_q1_a, q_target) + F.mse_loss(current_q2_a, q_target)
        opt_q.zero_grad()
        self.manual_backward(q_loss)
        opt_q.step()
        
        # 2. Update Value-network
        with torch.no_grad():
            t_q1 = self.target_q_network(obs)
            t_q2 = self.target_q_network2(obs)
            t_q = torch.min(t_q1, t_q2)
            t_q_a = t_q.gather(1, actions.unsqueeze(1)).view(-1)
            
        value = self.value_network(obs).view(-1)
        diff = t_q_a - value
        tau = self.get_cfg("tau", 0.7)
        weight = torch.where(diff > 0, tau, 1 - tau)
        value_loss = (weight * (diff**2)).mean()
        opt_v.zero_grad()
        self.manual_backward(value_loss)
        opt_v.step()
        
        # 3. Update Hybrid Actor
        with torch.no_grad():
            adv = t_q_a - value
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            beta = self.get_cfg("beta", 3.0)
            weights = torch.exp(beta * adv)
            weights = torch.clamp(weights, max=100.0)
            
        # Get logprobs from hybrid model
        _, log_probs, entropy, blend_entropy, _ = self.model.get_action_and_value(obs, logic_obs, actions)
        
        actor_loss = -(weights * log_probs).mean()
        # Add entropy regularization if desired
        actor_loss -= self.get_cfg("ent_coef", 0.0) * entropy.mean()
        actor_loss -= self.get_cfg("blend_ent_coef", 0.0) * blend_entropy.mean()
        
        opt_a.zero_grad()
        self.manual_backward(actor_loss)
        opt_a.step()
        
        self._soft_update(self.q_network, self.target_q_network)
        self._soft_update(self.q_network2, self.target_q_network2)
        
        self._log_offline_transitions()

        self.log_dict({
            "losses/q_loss": q_loss,
            "losses/value_loss": value_loss,
            "losses/actor_loss": actor_loss,
            "losses/entropy": entropy.mean(),
            "losses/blend_entropy": blend_entropy.mean(),
        })

    def configure_optimizers(self):
        # We use a standard LR from the agent config if available, otherwise fallback
        lr = self.cfg.agent.get("lr", 3e-4)
        opt_q = optim.Adam(list(self.q_network.parameters()) + list(self.q_network2.parameters()), lr=lr)
        opt_v = optim.Adam(self.value_network.parameters(), lr=lr)
        
        # Actor optimizer includes all heterogeneous policy modules and the blender
        actor_params = list(self.model.policy_modules.parameters()) + \
                       list(self.model.blender.parameters())
        opt_a = optim.Adam(actor_params, lr=lr)
        
        return [opt_q, opt_v, opt_a]

    def validation_step(self, batch, batch_idx):
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and getattr(datamodule, "val_reader", None) is not None:
            val_batch = datamodule.val_reader.get_batch(batch, device=self.device)
            obs = val_batch["obs"]
            if val_batch["logic_obs"] is not None:
                logic_obs = val_batch["logic_obs"]
            else:
                logic_obs = obs.unsqueeze(1).repeat(1, 2, 1)
            actions = val_batch["action"]
            rewards = val_batch["reward"]
            next_obs = val_batch["next_obs"]
            dones = val_batch["done"]
            
            with torch.no_grad():
                # Q-loss on validation
                target_v = self.value_network(next_obs).view(-1)
                q_target = rewards + self.cfg.env.gamma * target_v * (1 - dones)
                pred_q1 = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                pred_q2 = self.q_network2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                q_loss = F.mse_loss(pred_q1, q_target) + F.mse_loss(pred_q2, q_target)
                
                # Value loss on validation
                t_q1 = self.target_q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                t_q2 = self.target_q_network2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                t_q_a = torch.min(t_q1, t_q2)
                value = self.value_network(obs).view(-1)
                diff = t_q_a - value
                tau = self.get_cfg("tau", 0.7)
                weight = torch.where(diff > 0, tau, 1 - tau)
                value_loss = (weight * (diff**2)).mean()
                
                val_loss = q_loss + value_loss
                
            self.log("val/loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("val/q_loss", q_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
            self.log("val/value_loss", value_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
            return val_loss
