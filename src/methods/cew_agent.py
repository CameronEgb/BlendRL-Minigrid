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
        
        # Extract algorithm name
        def get_algo_name(acfg):
            if isinstance(acfg, (dict, DictConfig)):
                if "algorithm" in acfg: return acfg.algorithm
                if "agent" in acfg: return get_algo_name(acfg.agent)
                if "name" in acfg: return acfg.name
            return None
        self.algorithm = get_algo_name(cfg.agent)
        
        # Initialize placeholders for the fuzzy model
        self.fuzzy_model = None
        self.target_fuzzy_model = None
        self.rules = None
        self.antecedents = None
        self.self_organized = False
        
        self.automatic_optimization = False

    def on_train_epoch_start(self):
        # Base class handles datamodule reader limit
        super().on_train_epoch_start()
        
        epochs_per_interval = self.cfg.agent.get("epochs_per_interval", 1)
        if self.current_epoch % epochs_per_interval == 0:
            # Re-run self-organization at each interval boundary
            self.self_organize()

    def self_organize(self):
        print(f"Self-organizing for interval at epoch {self.current_epoch}...")
        datamodule = self.trainer.datamodule
        sample_size = min(len(datamodule.reader), 10000)
        batch = datamodule.reader.sample(sample_size)
        obs = batch["obs"].cpu().numpy()
        
        # 1. CLIP: Generate membership functions
        mins = obs.min(axis=0)
        maxes = obs.max(axis=0)
        self.antecedents = run_CLIP(obs, mins, maxes)
        
        # 2. ECM: Generate rule candidates (reduced centers)
        dthr = self.cfg.agent.get("ecm_dthr", 0.4)
        clusters = run_ECM(obs, [], dthr)
        reduced_X = np.array([c.center for c in clusters])
        
        # 3. WM: Generate rules
        self.rules = rule_creation(reduced_X, self.antecedents)
        
        # 4. FYD (if applicable)
        if "fyd" in self.algorithm:
            top_k = self.cfg.agent.get("fyd_top_k", None)
            self.rules = run_FYD(self.rules, obs, self.antecedents, top_k=top_k)
            
        # 5. Initialize/Re-initialize MIMO MultiFLC
        self.fuzzy_model = MultiFLC(
            n_inputs=obs.shape[1],
            n_outputs=self.n_actions,
            antecedents=self.antecedents,
            rules=self.rules,
            learning_rate=self.cfg.agent.lr,
            cql_alpha=self.cfg.agent.get("cql_alpha", 0.5)
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
        if not self.self_organized:
            return
            
        datamodule = self.trainer.datamodule
        cfg = self.cfg
        
        real_batch = datamodule.reader.sample(cfg.agent.batch_size)
        obs = real_batch["obs"]
        actions = real_batch["action"]
        rewards = real_batch["reward"]
        next_obs = real_batch["next_obs"]
        dones = real_batch["done"]
        
        # MIMO Forward pass (Batch, Actions)
        with torch.no_grad():
            next_q = self.target_fuzzy_model(next_obs)
            next_v = torch.max(next_q, dim=1)[0]
            q_target = rewards + cfg.env.gamma * next_v * (1 - dones)
            
        all_q_values = self.fuzzy_model(obs)
        
        logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
        q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        cql_loss = (logsumexp_qvalues - q_action).mean()
        bellman_loss = F.mse_loss(q_action, q_target)
        
        loss = bellman_loss + self.fuzzy_model.cql_alpha * cql_loss
        
        self.optimizer.zero_grad()
        self.manual_backward(loss)
        self.optimizer.step()
        
        self._soft_update_fuzzy(self.fuzzy_model, self.target_fuzzy_model)
        
        # Calculate current transitions for logging
        epochs_per_interval = self.cfg.agent.get("epochs_per_interval", 1)
        current_interval = self.current_epoch // epochs_per_interval
        interval_size = cfg.total_timesteps // cfg.intervals_count
        current_transitions = interval_size * (current_interval + 1)

        self.log_dict({
            "losses/total_loss": loss,
            "losses/bellman_loss": bellman_loss,
            "losses/cql_loss": cql_loss,
            "stats/n_rules": float(len(self.rules)),
            "stats/q_mean": all_q_values.mean(),
            "stats/q_max": all_q_values.max(),
        })
        self.log("transitions", float(current_transitions), logger=False, prog_bar=True)
        
        if batch_idx % 100 == 0:
            print(f"Epoch {self.current_epoch} Batch {batch_idx}: Loss={loss.item():.4f} Rules={len(self.rules)}")

    def _soft_update_fuzzy(self, model, target_model):
        tau = self.cfg.agent.get("soft_target_tau", 0.005)
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def configure_optimizers(self):
        # We handle optimizers manually, but Lightning needs something.
        # We'll re-init in self_organize anyway.
        return optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-4)

    def get_action_and_value(self, obs, logic_obs=None):
        if self.fuzzy_model is None:
            # Fallback for point 0 evaluation before first self-organize
            batch_size = obs.shape[0]
            return torch.zeros(batch_size, dtype=torch.long, device=self.device), \
                   torch.zeros(batch_size, device=self.device), \
                   torch.zeros(batch_size, device=self.device), \
                   torch.zeros(batch_size, device=self.device)
                   
        return self.fuzzy_model.get_action_and_value(obs)
