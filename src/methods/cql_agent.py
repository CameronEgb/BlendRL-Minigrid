import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict
from src.methods.base_agent import OfflineAgentBase
from src.methods.registry import register_agent

@register_agent("cql")
class CQLAgent(OfflineAgentBase):
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
        self.n_actions = self.env.n_actions if not callable(self.env.n_actions) else self.env.n_actions()
        
        hidden_sizes = cfg.agent.get("hidden_sizes", [64, 64])
        if hidden_sizes is not None:
            hidden_sizes = list(hidden_sizes)

        num_in_features = np.prod(self.observation_space)
        if cfg.env.architecture == "mlp":
            from src.utils import MLPQNetwork
            self.q_network = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features, hidden_sizes=hidden_sizes)
            self.target_q_network = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features, hidden_sizes=hidden_sizes)
        else:
            from src.utils import QNetwork
            self.q_network = QNetwork(n_actions=self.n_actions)
            self.target_q_network = QNetwork(n_actions=self.n_actions)
            
        from src.utils import get_neural_agent
        self.actor = get_neural_agent(cfg.env.name, self.n_actions, self.device, arch_name=cfg.env.architecture, hidden_sizes=hidden_sizes)
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        probs = self.actor.get_action_probs(obs)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        q_vals = self.q_network.get_q_values(obs)
        value = q_vals.max(dim=-1)[0]
        return action, logprob, entropy, value

    def get_value(self, obs, logic_obs=None):
        q_vals = self.q_network.get_q_values(obs)
        return q_vals.max(dim=-1)[0]

    def on_train_start(self):
        if hasattr(self.trainer.datamodule, "reader") and self.trainer.datamodule.reader is not None:
            self.trainer.datamodule.reader.device = self.device

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        datamodule = self.trainer.datamodule
        cfg = self.cfg
        
        # Sample a real batch from the reader
        real_batch = datamodule.reader.sample(cfg.agent.batch_size)
        obs = real_batch["obs"]
        actions = real_batch["action"]
        rewards = real_batch["reward"]
        next_obs = real_batch["next_obs"]
        dones = real_batch["done"]
        
        opt_q, opt_a = self.optimizers()
        
        # 1. Update Q-network using CQL loss
        with torch.no_grad():
            next_q = self.target_q_network(next_obs)
            next_v = torch.max(next_q, dim=1)[0]
            q_target = rewards + cfg.env.gamma * next_v * (1 - dones)
            
        all_q_values = self.q_network(obs)
        q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        bellman_loss = F.mse_loss(q_action, q_target)
        
        # CQL penalty component: logsumexp(Q) - Q(s,a)
        logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
        cql_alpha = self.get_cfg("cql_alpha", 1.0)
        cql_loss = (logsumexp_qvalues - q_action).mean()
        
        q_loss = bellman_loss + cql_alpha * cql_loss
        
        opt_q.zero_grad()
        self.manual_backward(q_loss)
        opt_q.step()
        
        # 2. Update Actor (expected Q maximization policy extraction)
        if hasattr(self.actor, "network") and hasattr(self.actor, "actor"):
            logits = self.actor.actor(self.actor.network(obs.float().reshape(obs.shape[0], -1)))
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            with torch.no_grad():
                q_vals = self.q_network(obs)
                
            actor_loss = (probs * (0.01 * log_probs - q_vals)).sum(dim=-1).mean()
        else:
            _, log_probs, entropy, _ = self.actor.get_action_and_value(obs, actions)
            with torch.no_grad():
                q_vals = self.q_network(obs)
                q_val_act = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
            actor_loss = -(q_val_act * log_probs).mean() - 0.01 * entropy.mean()
        
        opt_a.zero_grad()
        self.manual_backward(actor_loss)
        opt_a.step()
        
        self._soft_update(self.q_network, self.target_q_network)
        
        epochs_per_interval = self.cfg.agent.get("epochs_per_interval", 1)
        current_interval = self.current_epoch // epochs_per_interval
        interval_size = cfg.total_timesteps // cfg.intervals_count
        current_transitions = interval_size * (current_interval + 1)
        
        self.log_dict({
            "losses/q_loss": q_loss,
            "losses/bellman_loss": bellman_loss,
            "losses/cql_loss": cql_loss,
            "losses/actor_loss": actor_loss,
        })
        self.log("transitions", float(current_transitions), logger=False, prog_bar=True)

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        return self.actor.get_action_and_value(obs, action)

    def configure_optimizers(self):
        opt_q = optim.Adam(self.q_network.parameters(), lr=self.cfg.agent.lr)
        opt_a = optim.Adam(self.actor.parameters(), lr=self.cfg.agent.lr)
        return [opt_q, opt_a]
