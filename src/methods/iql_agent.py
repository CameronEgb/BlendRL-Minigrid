import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict, Optional
from src.methods.registry import register_agent
from src.methods.base_agent import OfflineAgentBase

@register_agent("iql")
class IQLAgent(OfflineAgentBase):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        
        # Handle nested agent config for algorithm
        algorithm = self.get_cfg("algorithm", self.get_cfg("name", cfg.env.name))

        # In offline mode, env is only for evaluation
        from blendrl.env_vectorized import VectorizedNudgeBaseEnv
        self.env = VectorizedNudgeBaseEnv.from_name(
            cfg.env.name, 
            n_envs=1, 
            mode=algorithm, 
            seed=cfg.seed
        )
        dummy_logic, dummy_neural = self.env.reset()
        self.observation_space = dummy_neural.shape[1:]
        self.n_actions = self.env.n_actions if not callable(self.env.n_actions) else self.env.n_actions()
        
        hidden_sizes = cfg.agent.get("hidden_sizes", [64, 64])
        if hidden_sizes is not None:
            hidden_sizes = list(hidden_sizes)

        # Initialize networks
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
        
        from src.utils import get_neural_agent
        self.actor = get_neural_agent(cfg.env.name, self.n_actions, self.device, arch_name=cfg.env.architecture, hidden_sizes=hidden_sizes)
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())

    def on_train_start(self):
        if hasattr(self.trainer.datamodule, "reader") and self.trainer.datamodule.reader is not None:
            self.trainer.datamodule.reader.device = self.device

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        # batch is from DataLoader, but we use the reader from datamodule
        datamodule = self.trainer.datamodule
        cfg = self.cfg
        
        # Sample a real batch from the reader
        real_batch = datamodule.reader.sample(cfg.agent.batch_size)
        obs = real_batch["obs"]
        actions = real_batch["action"]
        rewards = real_batch["reward"]
        next_obs = real_batch["next_obs"]
        dones = real_batch["done"]
        
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
        weight = torch.where(diff > 0, cfg.agent.tau, 1 - cfg.agent.tau)
        value_loss = (weight * (diff**2)).mean()
        opt_v.zero_grad()
        self.manual_backward(value_loss)
        opt_v.step()
        
        # 3. Update Actor
        with torch.no_grad():
            adv = t_q_a - value
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            weights = torch.exp(cfg.agent.beta * adv)
            weights = torch.clamp(weights, max=100.0)
            
        _, log_probs, _, _ = self.actor.get_action_and_value(obs, actions)
        actor_loss = -(weights * log_probs).mean()
        opt_a.zero_grad()
        self.manual_backward(actor_loss)
        opt_a.step()
        
        # Soft update target networks
        self._soft_update(self.q_network, self.target_q_network)
        self._soft_update(self.q_network2, self.target_q_network2)
        
        self._log_offline_transitions()

        self.log_dict({
            "losses/q_loss": q_loss,
            "losses/value_loss": value_loss,
            "losses/actor_loss": actor_loss,
        })

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        return self.actor.get_action_and_value(obs, action)

    def get_value(self, obs):
        return self.value_network(obs)

    def configure_optimizers(self):
        opt_q = optim.Adam(list(self.q_network.parameters()) + list(self.q_network2.parameters()), lr=self.cfg.agent.lr)
        opt_v = optim.Adam(self.value_network.parameters(), lr=self.cfg.agent.lr)
        opt_a = optim.Adam(self.actor.parameters(), lr=self.cfg.agent.lr)
        return [opt_q, opt_v, opt_a]

    def validation_step(self, batch, batch_idx):
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and getattr(datamodule, "val_reader", None) is not None:
            val_batch = datamodule.val_reader.get_batch(batch, device=self.device)
            obs = val_batch["obs"]
            actions = val_batch["action"]
            rewards = val_batch["reward"]
            next_obs = val_batch["next_obs"]
            dones = val_batch["done"]
            
            with torch.no_grad():
                # Value loss on validation
                q1 = self.target_q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                q2 = self.target_q_network2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                target_v = torch.min(q1, q2)
                v = self.value_network(obs).squeeze(-1)
                u = target_v - v
                expectile = self.get_cfg("expectile", 0.7)
                weight = torch.where(u > 0, expectile, 1 - expectile)
                value_loss = (weight * (u ** 2)).mean()
                
                # Q loss on validation
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
