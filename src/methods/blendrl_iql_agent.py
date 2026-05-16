import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict, Optional
from src.methods.iql_agent import IQLAgent
from blendrl.agents.blender_agent import BlenderActorCritic

class BlendRLIQLAgent(IQLAgent):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Handle nested agent config for algorithm
        agent_cfg = cfg.agent
        if "agent" in agent_cfg:
            algorithm = agent_cfg.agent.algorithm
        else:
            algorithm = agent_cfg.algorithm

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
        )

        # Q and Value networks are still neural-only in their standard implementation
        # but we could potentially make them hybrid too. For now, keep them neural.
        # However, the actor is the hybrid part.
        
        self.automatic_optimization = False

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        datamodule = self.trainer.datamodule
        cfg = self.cfg
        
        real_batch = datamodule.reader.sample(cfg.agent.batch_size)
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
        weight = torch.where(diff > 0, cfg.agent.tau, 1 - cfg.agent.tau)
        value_loss = (weight * (diff**2)).mean()
        opt_v.zero_grad()
        self.manual_backward(value_loss)
        opt_v.step()
        
        # 3. Update Hybrid Actor
        with torch.no_grad():
            adv = t_q_a - value
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            weights = torch.exp(cfg.agent.beta * adv)
            weights = torch.clamp(weights, max=100.0)
            
        # Get logprobs from hybrid model
        _, log_probs, entropy, blend_entropy, _ = self.model.get_action_and_value(obs, logic_obs, actions)
        
        actor_loss = -(weights * log_probs).mean()
        # Add entropy regularization if desired
        actor_loss -= cfg.agent.get("ent_coef", 0.0) * entropy.mean()
        actor_loss -= cfg.agent.get("blend_ent_coef", 0.0) * blend_entropy.mean()
        
        opt_a.zero_grad()
        self.manual_backward(actor_loss)
        opt_a.step()
        
        self._soft_update(self.q_network, self.target_q_network)
        self._soft_update(self.q_network2, self.target_q_network2)
        
        # Calculate current transitions for logging
        epochs_per_interval = self.cfg.agent.get("epochs_per_interval", 1)
        current_interval = self.current_epoch // epochs_per_interval
        interval_size = cfg.total_timesteps // cfg.intervals_count
        current_transitions = interval_size * (current_interval + 1)

        self.log_dict({
            "losses/q_loss": q_loss,
            "losses/value_loss": value_loss,
            "losses/actor_loss": actor_loss,
            "losses/entropy": entropy.mean(),
            "losses/blend_entropy": blend_entropy.mean(),
        })
        self.log("transitions", float(current_transitions), logger=False, prog_bar=True)

    def configure_optimizers(self):
        opt_q = optim.Adam(list(self.q_network.parameters()) + list(self.q_network2.parameters()), lr=self.cfg.agent.lr)
        opt_v = optim.Adam(self.value_network.parameters(), lr=self.cfg.agent.lr)
        
        # Actor optimizer includes neural actor, logic actor, and blender
        actor_params = list(self.model.visual_neural_actor.parameters()) + \
                       list(self.model.logic_actor.parameters()) + \
                       list(self.model.blender.parameters())
        opt_a = optim.Adam(actor_params, lr=self.cfg.agent.lr)
        
        return [opt_q, opt_v, opt_a]

    def validation_step(self, batch, batch_idx):
        # validation_step placeholder
        pass
