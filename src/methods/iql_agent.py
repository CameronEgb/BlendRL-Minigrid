import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict, Optional

from src.methods.registry import register_agent
from src.methods.base_agent import OfflineAgentBase
from blendrl.agents.blender_agent import BlenderActorCritic


@register_agent(
    "iql",
    "iql_dnn",
    "blendrl_iql",
    "iql_blendrl_human_neural",
    "blendrl_iql_human_neural",
)
class IQLAgent(OfflineAgentBase):
    """Unified Implicit Q-Learning (IQL) Offline RL Agent.
    
    Supports pure neural architectures as well as hybrid BlendRL modular actor policies.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        
        self._init_env(n_envs=1)
        
        hidden_sizes = cfg.agent.get("hidden_sizes", [256, 256])
        if hidden_sizes is not None:
            hidden_sizes = list(hidden_sizes)

        num_in_features = np.prod(self.observation_space)
        if cfg.env.architecture == "mlp":
            from src.utils import MLPQNetwork, MLPValueNetwork
            self.q_network = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features, hidden_sizes=hidden_sizes)
            self.q_network2 = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features, hidden_sizes=hidden_sizes)
            self.value_network = MLPValueNetwork(num_in_features=num_in_features, hidden_sizes=hidden_sizes)
            
            self.target_q_network = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features, hidden_sizes=hidden_sizes)
            self.target_q_network2 = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features, hidden_sizes=hidden_sizes)
        else:
            from src.utils import QNetwork, ValueNetwork
            self.q_network = QNetwork(n_actions=self.n_actions)
            self.q_network2 = QNetwork(n_actions=self.n_actions)
            self.value_network = ValueNetwork()
            
            self.target_q_network = QNetwork(n_actions=self.n_actions)
            self.target_q_network2 = QNetwork(n_actions=self.n_actions)

        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())

        # Check if modular/hybrid actor is configured
        has_modules = bool(self.get_cfg("modules", []))
        is_hybrid = self.get_cfg("actor_mode", "neural") in ["hybrid", "logic"] or "blendrl" in str(algorithm)
        self.is_modular = has_modules or is_hybrid

        if self.is_modular:
            self.model = BlenderActorCritic(
                self.env,
                self.get_cfg("rules", cfg.env.rules),
                self.get_cfg("actor_mode", "hybrid"),
                self.get_cfg("blender_mode", "neural"),
                self.get_cfg("blend_function", "softmax"),
                self.get_cfg("reasoner", cfg.env.reasoner),
                self.device,
                architecture=self.get_cfg("architecture", cfg.env.architecture),
                cfg=cfg.agent
            )
        else:
            from src.utils import get_neural_agent
            self.actor = get_neural_agent(cfg.env.name, self.n_actions, self.device, arch_name=cfg.env.architecture, hidden_sizes=hidden_sizes)

    def _prepare_logic_obs(self, obs, logic_obs=None):
        if logic_obs is not None:
            return logic_obs.to(self.device)
        if obs.ndim == 2:
            return obs.unsqueeze(1).repeat(1, 2, 1).to(self.device)
        return obs.to(self.device)

    def on_train_start(self):
        if hasattr(self.trainer.datamodule, "reader") and self.trainer.datamodule.reader is not None:
            self.trainer.datamodule.reader.device = self.device

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.is_modular and hasattr(self.model, "self_organize_cew_modules"):
            epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
            if self.current_epoch % epochs_per_interval == 0:
                datamodule = self.trainer.datamodule
                sample_size = min(len(datamodule.reader), 10000)
                if sample_size > 0:
                    batch = datamodule.reader.sample(sample_size)
                    organize_obs = batch["logic_obs"] if batch["logic_obs"] is not None else batch["obs"]
                    if self.model.self_organize_cew_modules(organize_obs):
                        lr = self.get_cfg("lr", 3e-4)
                        actor_params = list(self.model.policy_modules.parameters()) + list(self.model.blender.parameters())
                        self.trainer.strategy.optimizers[2] = optim.Adam(actor_params, lr=lr)

    def training_step(self, batch, batch_idx):
        datamodule = getattr(self.trainer, "datamodule", None)
        cfg = self.cfg
        
        if isinstance(batch, dict) and "obs" in batch:
            real_batch = batch
        elif datamodule is not None and getattr(datamodule, "reader", None) is not None:
            if isinstance(batch, torch.Tensor):
                real_batch = datamodule.reader.get_batch(batch, device=self.device)
            else:
                batch_size = self.get_cfg("batch_size", 256)
                real_batch = datamodule.reader.sample(batch_size)
        else:
            raise RuntimeError("IQLAgent requires an active offline dataset reader or batched dictionary.")
            
        obs = real_batch["obs"].to(self.device, non_blocking=True)
        actions = real_batch["action"].to(self.device, non_blocking=True)
        rewards = real_batch["reward"].to(self.device, non_blocking=True)
        next_obs = real_batch["next_obs"].to(self.device, non_blocking=True)
        dones = real_batch["done"].to(self.device, non_blocking=True)
        
        opt_q, opt_v, opt_a = self.optimizers()
        
        # 1. Update Q-networks
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
        tau = self.get_cfg("tau", self.get_cfg("expectile", 0.7))
        weight = torch.where(diff > 0, tau, 1 - tau)
        value_loss = (weight * (diff**2)).mean()
        opt_v.zero_grad()
        self.manual_backward(value_loss)
        opt_v.step()
        
        # 3. Update Actor
        with torch.no_grad():
            adv = t_q_a - value
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            beta = self.get_cfg("beta", self.get_cfg("temperature", 3.0))
            weights = torch.exp(beta * adv)
            weights = torch.clamp(weights, max=100.0)
            
        if self.is_modular:
            logic_obs = self._prepare_logic_obs(obs, real_batch.get("logic_obs"))
            _, log_probs, _, blend_entropy, _ = self.model(obs, logic_obs, action=actions)
        else:
            _, log_probs, _, _ = self.actor.get_action_and_value(obs, actions)
            blend_entropy = None
            
        actor_loss = -(weights * log_probs).mean()
        if self.is_modular and isinstance(blend_entropy, torch.Tensor):
            blend_ent_coef = self.get_cfg("blend_ent_coef", 0.01)
            actor_loss = actor_loss - blend_ent_coef * blend_entropy.mean()
        opt_a.zero_grad()
        self.manual_backward(actor_loss)
        opt_a.step()
        
        # Soft update target networks
        soft_target_tau = self.get_cfg("soft_target_tau", 0.005)
        self._soft_update(self.q_network, self.target_q_network, tau=soft_target_tau)
        self._soft_update(self.q_network2, self.target_q_network2, tau=soft_target_tau)
        
        self._log_offline_transitions()

        log_data = {
            "losses/q_loss": q_loss,
            "losses/value_loss": value_loss,
            "losses/actor_loss": actor_loss,
        }
        if self.is_modular and isinstance(blend_entropy, torch.Tensor):
            log_data["losses/blend_entropy"] = blend_entropy.mean().item()
        self.log_dict(log_data)

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        if self.is_modular:
            logic_obs = self._prepare_logic_obs(obs, logic_obs)
            return self.model.get_action_and_value(obs, logic_obs, action)
        return self.actor.get_action_and_value(obs, action)

    def get_value(self, obs, logic_obs=None):
        return self.value_network(obs)

    def configure_optimizers(self):
        opt_q = optim.Adam(list(self.q_network.parameters()) + list(self.q_network2.parameters()), lr=self.cfg.agent.lr)
        opt_v = optim.Adam(self.value_network.parameters(), lr=self.cfg.agent.lr)
        
        if self.is_modular:
            actor_params = list(self.model.policy_modules.parameters()) + list(self.model.blender.parameters())
            opt_a = optim.Adam(actor_params, lr=self.cfg.agent.lr)
        else:
            opt_a = optim.Adam(self.actor.parameters(), lr=self.cfg.agent.lr)
        return [opt_q, opt_v, opt_a]

    def validation_step(self, batch, batch_idx):
        datamodule = getattr(self.trainer, "datamodule", None)
        if isinstance(batch, dict) and "obs" in batch:
            val_batch = batch
        elif datamodule is not None and getattr(datamodule, "val_reader", None) is not None:
            val_batch = datamodule.val_reader.get_batch(batch, device=self.device)
        else:
            return

        obs = val_batch["obs"].to(self.device, non_blocking=True)
        actions = val_batch["action"].to(self.device, non_blocking=True)
        rewards = val_batch["reward"].to(self.device, non_blocking=True)
        next_obs = val_batch["next_obs"].to(self.device, non_blocking=True)
        dones = val_batch["done"].to(self.device, non_blocking=True)
            
        with torch.no_grad():
            q1 = self.target_q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            q2 = self.target_q_network2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            target_v = torch.min(q1, q2)
            v = self.value_network(obs).squeeze(-1)
            u = target_v - v
            expectile = self.get_cfg("tau", self.get_cfg("expectile", 0.7))
            weight = torch.where(u > 0, expectile, 1 - expectile)
            value_loss = (weight * (u ** 2)).mean()
            
            next_v = self.value_network(next_obs).squeeze(-1)
            q_target = rewards + self.cfg.env.gamma * next_v * (1 - dones)
            pred_q1 = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            pred_q2 = self.q_network2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(pred_q1, q_target) + F.mse_loss(pred_q2, q_target)
            
            val_loss = value_loss + q_loss
            
        self.log("val/loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/q_loss", q_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/value_loss", value_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        return val_loss
