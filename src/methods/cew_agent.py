import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict, Optional
from omegaconf import DictConfig
from src.methods.iql_agent import IQLAgent
from src.methods.cew_utils import run_CLIP, run_ECM, rule_creation, run_FYD, MultiFLC

class CEWAgent(IQLAgent):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        self.cfg = cfg
        
        def get_algo_name(acfg):
            if isinstance(acfg, (dict, DictConfig)):
                if "algorithm" in acfg: return acfg.algorithm
                if "agent" in acfg: return get_algo_name(acfg.agent)
                if "name" in acfg: return acfg.name
            return None
        self.algorithm = get_algo_name(cfg.agent)
        
        self.fuzzy_model = None
        self.target_fuzzy_model = None
        self.rules = None
        self.antecedents = None
        self.self_organized = False
        self.automatic_optimization = False
        
        self.obs_min = None
        self.obs_max = None

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        epochs_per_interval = self.cfg.agent.get("epochs_per_interval", 1)
        if self.current_epoch % epochs_per_interval == 0:
            self.self_organize()

    def normalize_obs(self, obs):
        return obs # No normalization

    def self_organize(self):
        print(f"Self-organizing for interval at epoch {self.current_epoch}...")
        datamodule = self.trainer.datamodule
        
        # Use last 20k transitions for self-org
        sample_size = min(len(datamodule.reader), 20000)
        batch = datamodule.reader.sample(sample_size, last=True)
        obs = batch["obs"].cpu().numpy()
        
        # 1. CLIP
        eps = self.cfg.agent.get("eps", 0.2)
        kappa = self.cfg.agent.get("kappa", 0.6)
        self.antecedents = run_CLIP(obs, obs.min(axis=0), obs.max(axis=0), eps=eps, kappa=kappa)
        
        # 2. ECM
        dthr = self.cfg.agent.get("ecm_dthr", 0.1)
        clusters = run_ECM(obs, [], dthr)
        reduced_X = np.array([c.center for c in clusters])
        
        # 3. WM
        self.antecedents, self.rules = rule_creation(reduced_X, self.antecedents)
        
        # 4. FYD
        if "fyd" in self.algorithm:
            top_k = self.cfg.agent.get("fyd_top_k", None)
            self.rules = run_FYD(self.rules, obs, self.antecedents, top_k=top_k)
            
        # 5. Initialize MultiFLC
        self.fuzzy_model = MultiFLC(
            n_inputs=obs.shape[1],
            n_outputs=self.n_actions,
            antecedents=self.antecedents,
            rules=self.rules,
            learning_rate=self.cfg.agent.lr,
            cql_alpha=self.cfg.agent.get("cql_alpha", 1.0)
        ).to(self.device)
        
        self.target_fuzzy_model = MultiFLC(
            n_inputs=obs.shape[1],
            n_outputs=self.n_actions,
            antecedents=self.antecedents,
            rules=self.rules
        ).to(self.device)
        self.target_fuzzy_model.load_state_dict(self.fuzzy_model.state_dict())
        
        self.optimizer = optim.Adam(self.fuzzy_model.parameters(), lr=self.cfg.agent.lr)
        self.self_organized = True
        print(f"Self-organization complete. MIMO Rules: {len(self.rules)}")

    def training_step(self, batch, batch_idx):
        if not self.self_organized: return
            
        datamodule = self.trainer.datamodule
        real_batch = datamodule.reader.sample(self.cfg.agent.batch_size)
        obs = self.normalize_obs(real_batch["obs"])
        actions = real_batch["action"]
        rewards = real_batch["reward"]
        next_obs = self.normalize_obs(real_batch["next_obs"])
        dones = real_batch["done"]
        
        with torch.no_grad():
            next_q = self.target_fuzzy_model(next_obs)
            next_v = torch.max(next_q, dim=1)[0]
            q_target = rewards + self.cfg.env.gamma * next_v * (1 - dones)
            
        all_q_values = self.fuzzy_model(obs)
        q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
        cql_loss = (logsumexp_qvalues - q_action).mean()
        bellman_loss = F.mse_loss(q_action, q_target)
        
        loss = bellman_loss + self.fuzzy_model.cql_alpha * cql_loss
        
        self.optimizer.zero_grad()
        self.manual_backward(loss)
        self.optimizer.step()
        self._soft_update_fuzzy(self.fuzzy_model, self.target_fuzzy_model)
        
        if batch_idx % 10 == 0:
            weight_norm = torch.cat([p.flatten() for p in self.fuzzy_model.parameters()]).norm().item()
            print(f"Epoch {self.current_epoch} Batch {batch_idx}: Loss={loss.item():.4f} Q_mean={all_q_values.mean().item():.4f} W_norm={weight_norm:.4f}")
        
        # Calculate current transitions for logging
        epochs_per_interval = self.cfg.agent.get("epochs_per_interval", 1)
        current_interval = self.current_epoch // epochs_per_interval
        interval_size = self.cfg.total_timesteps // self.cfg.intervals_count
        current_transitions = interval_size * (current_interval + 1)

        self.log_dict({
            "losses/total_loss": loss,
            "losses/bellman_loss": bellman_loss,
            "losses/cql_loss": cql_loss,
            "train/q_mean": all_q_values.mean(),
            "train/rules": float(len(self.rules))
        })
        self.log("transitions", float(current_transitions), logger=False, prog_bar=True)
        
        if batch_idx % 100 == 0:
            print(f"Epoch {self.current_epoch} Batch {batch_idx}: Loss={loss.item():.4f} Rules={len(self.rules)} Q_mean={all_q_values.mean().item():.4f}")

    def _soft_update_fuzzy(self, model, target_model):
        tau = self.cfg.agent.get("soft_target_tau", 0.005)
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def configure_optimizers(self):
        return optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-4)

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        if self.fuzzy_model is None:
            batch_size = obs.shape[0]
            return torch.zeros(batch_size, dtype=torch.long, device=self.device), \
                   torch.zeros(batch_size, device=self.device), \
                   torch.zeros(batch_size, device=self.device), \
                   torch.zeros(batch_size, device=self.device)
        
        act, log_p, ent, val = self.fuzzy_model.get_action_and_value(self.normalize_obs(obs))
        if np.random.random() < 0.01: # Print 1% of steps
             print(f"DEBUG: Actions={act[:5].tolist()} Q-Values Mean={val.mean().item():.4f}")
        return act, log_p, ent, val
