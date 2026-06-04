import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict, Optional
from src.methods.ppo_agent import PPOAgent

class IQLAgent(PPOAgent):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Handle nested agent config for algorithm
        algorithm = self.get_cfg("algorithm", self.get_cfg("name", cfg.env.name))

        # In offline mode, env is only for evaluation
        from src.blendrl.env_vectorized import VectorizedNudgeBaseEnv
        self.env = VectorizedNudgeBaseEnv.from_name(
            cfg.env.name, 
            n_envs=1, 
            mode=algorithm, 
            seed=cfg.seed
        )
        dummy_logic, dummy_neural = self.env.reset()
        self.observation_space = dummy_neural.shape[1:]
        self.n_actions = self.env.n_actions if not callable(self.env.n_actions) else self.env.n_actions()
        
        # Initialize networks
        num_in_features = np.prod(self.observation_space)
        if cfg.env.architecture == "mlp":
            from src.utils import MLPQNetwork, MLPValueNetwork
            self.q_network = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features)
            self.q_network2 = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features)
            self.value_network = MLPValueNetwork(num_in_features=num_in_features)
            
            self.target_q_network = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features)
            self.target_q_network2 = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features)
        else:
            from src.utils import QNetwork, ValueNetwork
            self.q_network = QNetwork(n_actions=self.n_actions)
            self.q_network2 = QNetwork(n_actions=self.n_actions)
            self.value_network = ValueNetwork()
            
            self.target_q_network = QNetwork(n_actions=self.n_actions)
            self.target_q_network2 = QNetwork(n_actions=self.n_actions)
        
        from src.utils import get_neural_agent
        self.actor = get_neural_agent(cfg.env.name, self.n_actions, self.device, arch_name=cfg.env.architecture)
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())
        
        self.automatic_optimization = False

    def on_train_start(self):
        if hasattr(self.trainer.datamodule, "reader") and self.trainer.datamodule.reader is not None:
            self.trainer.datamodule.reader.device = self.device

    def on_train_epoch_start(self):
        if self.cfg.mode.type == "offline":
            datamodule = self.trainer.datamodule
            if hasattr(datamodule, "reader") and datamodule.reader is not None:
                # Calculate limit based on current interval
                # Intervals are blocks of epochs_per_interval
                epochs_per_interval = self.cfg.agent.get("epochs_per_interval", 1)
                current_interval = self.current_epoch // epochs_per_interval
                
                interval_size = self.cfg.total_timesteps // self.cfg.intervals_count
                current_limit = interval_size * (current_interval + 1)
                datamodule.reader.set_limit(current_limit)

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
        
        # Calculate current transitions for logging
        epochs_per_interval = self.cfg.agent.get("epochs_per_interval", 1)
        current_interval = self.current_epoch // epochs_per_interval
        interval_size = cfg.total_timesteps // cfg.intervals_count
        current_transitions = interval_size * (current_interval + 1)

        self.log_dict({
            "losses/q_loss": q_loss,
            "losses/value_loss": value_loss,
            "losses/actor_loss": actor_loss,
        })
        self.log("transitions", float(current_transitions), logger=False, prog_bar=True)

    def _soft_update(self, model, target_model):
        tau = self.cfg.agent.get("soft_target_tau", 0.005)
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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
        # We can use this to log validation loss on the offline dataset
        pass
