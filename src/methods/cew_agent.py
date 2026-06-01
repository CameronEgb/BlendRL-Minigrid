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
        # Sample some data from the current limited dataset to organize rules
        # Use a reasonably large sample for organization
        sample_size = min(len(datamodule.reader), 10000)
        batch = datamodule.reader.sample(sample_size)
        obs = batch["obs"].cpu().numpy()
        
        # 1. CLIP: Generate membership functions
        # Use full dataset range for CLIP initialization if possible, or current sample
        mins = obs.min(axis=0)
        maxes = obs.max(axis=0)
        self.antecedents = run_CLIP(obs, mins, maxes)
        
        # 2. ECM: Generate rule candidates (reduced centers)
        dthr = self.cfg.agent.get("ecm_dthr", 0.05)
        clusters = run_ECM(obs, [], dthr)
        reduced_X = np.array([c.center for c in clusters])
        
        # 3. WM: Generate rules
        self.rules = rule_creation(reduced_X, self.antecedents)
        
        # 4. FYD (if applicable)
        if "fyd" in self.algorithm:
            top_k = self.cfg.agent.get("fyd_top_k", None)
            self.rules = run_FYD(self.rules, obs, self.antecedents, top_k=top_k)
            
        # 5. Initialize/Re-initialize MultiFLC
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
        
        # Re-initialize optimizers for the new flcs
        self.optimizers_list = [optim.Adam(flc.parameters(), lr=self.cfg.agent.lr) for flc in self.fuzzy_model.flcs]
        
        self.self_organized = True
        print(f"Self-organization complete. Number of rules: {len(self.rules)}")

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
        
        # Update Target Q
        with torch.no_grad():
            next_q = self.target_fuzzy_model(next_obs)
            next_v = torch.max(next_q, dim=1)[0]
            q_target = rewards + cfg.env.gamma * next_v * (1 - dones)
            
        total_loss_val = 0
        for flc_idx, flc in enumerate(self.fuzzy_model.flcs):
            opt = self.optimizers_list[flc_idx]
            
            # Recompute all_q_values for each flc update to avoid graph issues
            # and ensure we use the latest values if we were doing sequential updates.
            # However, for efficiency, let's just use the current flc and detach others if possible.
            # Or simpler: compute all_q_values once with gradients, then update each flc.
            
            # Recompute for each to be safe with individual optimizers
            all_q_values = self.fuzzy_model(obs)
            pred_q = all_q_values[:, flc_idx]
            
            logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
            q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            cql_loss = (logsumexp_qvalues - q_action).mean()
            
            indices = (actions == flc_idx).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                bellman_loss = F.mse_loss(pred_q[indices], q_target[indices])
            else:
                bellman_loss = 0
                
            loss = bellman_loss + self.fuzzy_model.cql_alpha * cql_loss
            
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            total_loss_val += loss.item()
            
        # Soft update target
        self._soft_update_fuzzy(self.fuzzy_model, self.target_fuzzy_model)
        
        # Calculate current transitions for logging
        epochs_per_interval = self.cfg.agent.get("epochs_per_interval", 1)
        current_interval = self.current_epoch // epochs_per_interval
        interval_size = cfg.total_timesteps // cfg.intervals_count
        current_transitions = interval_size * (current_interval + 1)

        self.log("losses/total_loss", total_loss_val / len(self.fuzzy_model.flcs))
        self.log("transitions", float(current_transitions), logger=False, prog_bar=True)

    def _soft_update_fuzzy(self, model, target_model):
        tau = self.cfg.agent.get("soft_target_tau", 0.005)
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def configure_optimizers(self):
        # We handle optimizers manually in training_step because they are per-action FLC
        # But Lightning might want at least one or it might complain.
        # We'll return a dummy one and use manual optimization.
        return optim.Adam([torch.zeros(1, requires_grad=True)], lr=1.0)

    def get_action_and_value(self, obs, logic_obs=None):
        if self.fuzzy_model is None:
            # Fallback for point 0 evaluation before first self-organize
            batch_size = obs.shape[0]
            return torch.zeros(batch_size, dtype=torch.long, device=self.device), \
                   torch.zeros(batch_size, device=self.device), \
                   torch.zeros(batch_size, device=self.device), \
                   torch.zeros(batch_size, device=self.device)
                   
        return self.fuzzy_model.get_action_and_value(obs)
