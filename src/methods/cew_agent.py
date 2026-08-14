import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict, Optional
from omegaconf import DictConfig
from src.methods.cew_utils import run_CLIP, run_ECM, rule_creation, run_FYD, MultiFLC, stabilize_antecedents
from src.methods.registry import register_agent
from src.methods.base_agent import OfflineAgentBase

@register_agent("cew")
class CEWAgent(OfflineAgentBase):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        
        # Helper to get config values
        self.lr = self.get_cfg("lr", 3e-4)
        self.algorithm = self.get_cfg("algorithm", self.get_cfg("name", "cew"))
        
        # Initialize internal state
        self.fuzzy_model = None
        self.target_fuzzy_model = None
        self.rules = None
        self.antecedents = None
        self.self_organized = False
        
        # Setup for evaluation
        from blendrl.env_vectorized import VectorizedNudgeBaseEnv
        self.eval_env = VectorizedNudgeBaseEnv.from_name(
            cfg.env.name, 
            n_envs=1, 
            mode=self.algorithm, 
            seed=cfg.seed
        )
        _, dummy_neural = self.eval_env.reset()
        self.observation_space = dummy_neural.shape[1:]
        self.n_actions = self.eval_env.n_actions if not callable(self.eval_env.n_actions) else self.eval_env.n_actions()

    def on_train_epoch_start(self):
        """Handle interval-based dataset scaling and self-organization."""
        super().on_train_epoch_start()
        
        epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
        # Re-run self-organization at the start of each interval
        if self.current_epoch % epochs_per_interval == 0:
            self.self_organize()

    def self_organize(self):
        """Isomorphic implementation of the CEW/FYD self-organization pipeline."""
        print(f"Self-organizing for interval at epoch {self.current_epoch}...")
        datamodule = self.trainer.datamodule
        sample_size = min(len(datamodule.reader), 20000)
        batch = datamodule.reader.sample(sample_size, last=True)
        obs = batch["obs"].cpu().numpy()
        
        # 1. CLIP: Categorical Learning Induced Partitioning
        eps = self.get_cfg("eps", 0.1)
        kappa = self.get_cfg("kappa", 0.6)
        new_antecedents = run_CLIP(obs, obs.min(axis=0), obs.max(axis=0), eps=eps, kappa=kappa)
        
        # 2. ECM: Evolving Clustering Method
        dthr = self.get_cfg("ecm_dthr", 0.1)
        clusters = run_ECM(obs, [], dthr)
        reduced_X = np.array([c.center for c in clusters])
        
        # 3. WM: Wang-Mendel Rule Creation
        new_antecedents, new_rules = rule_creation(reduced_X, new_antecedents)
        
        # 4. Stabilization: Mamdani Autoencoder (Refining fuzzy sets)
        if self.get_cfg("stabilize", True):
            new_antecedents = stabilize_antecedents(
                obs, new_antecedents, new_rules, "cpu",
                lr=self.get_cfg("stabilize_lr", 1e-3),
                epochs=self.get_cfg("stabilize_epochs", 10)
            )
        
        # 5. FYD: Frequent-Yet-Discernible (Pruning)
        if "fyd" in self.algorithm:
            top_k = self.get_cfg("fyd_top_k", None)
            new_rules, new_antecedents = run_FYD(new_rules, obs, new_antecedents, top_k=top_k)
            
        # Check if architecture changed before resetting weights
        if self.fuzzy_model is not None:
            # Check rule count and antecedent count
            if len(new_rules) == len(self.rules):
                current_ant_count = sum(len(p_ants) for p_ants in self.antecedents)
                new_ant_count = sum(len(p_ants) for p_ants in new_antecedents)
                if current_ant_count == new_ant_count:
                    print(f"CEW architecture stable ({len(new_rules)} rules). Skipping reset.")
                    return

        self.rules = new_rules
        self.antecedents = new_antecedents

        # 6. Initialize MultiFLC (MIMO architecture: one FLC per action)
        self.fuzzy_model = MultiFLC(
            n_inputs=obs.shape[1],
            n_outputs=self.n_actions,
            antecedents=self.antecedents,
            rules=self.rules,
            learning_rate=self.lr,
            cql_alpha=self.get_cfg("cql_alpha", 1.0)
        ).to("cpu") # Keep fuzzy logic on CPU
        
        self.target_fuzzy_model = MultiFLC(
            n_inputs=obs.shape[1],
            n_outputs=self.n_actions,
            antecedents=self.antecedents,
            rules=self.rules
        ).to("cpu")
        self.target_fuzzy_model.load_state_dict(self.fuzzy_model.state_dict())
        
        self.opt_fuzzy = optim.Adam(self.fuzzy_model.parameters(), lr=self.lr)
        self.self_organized = True
        print(f"Self-organization complete. MIMO Rules: {len(self.rules)}. Weights reset.")

    def training_step(self, batch, batch_idx):
        if not self.self_organized: return
            
        datamodule = self.trainer.datamodule
        real_batch = datamodule.reader.sample(self.get_cfg("batch_size", 32))
        obs = real_batch["obs"].to("cpu")
        actions = real_batch["action"].to("cpu")
        rewards = real_batch["reward"].to("cpu")
        next_obs = real_batch["next_obs"].to("cpu")
        dones = real_batch["done"].to("cpu")
        
        with torch.no_grad():
            next_q = self.target_fuzzy_model(next_obs)
            next_v = torch.max(next_q, dim=1)[0]
            q_target = rewards + self.get_cfg("gamma", 0.99) * next_v * (1 - dones)
            
        all_q_values = self.fuzzy_model(obs)
        q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # CQL component: logsumexp(Q) - Q(s,a)
        logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
        cql_loss = (logsumexp_qvalues - q_action).mean()
        
        bellman_loss = F.mse_loss(q_action, q_target)
        
        # Entropy bonus
        probs = torch.softmax(all_q_values, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
        
        total_loss = bellman_loss + self.fuzzy_model.cql_alpha * cql_loss - 0.01 * entropy
        
        self.opt_fuzzy.zero_grad()
        self.manual_backward(total_loss)
        self.opt_fuzzy.step()
        
        self._soft_update(self.fuzzy_model, self.target_fuzzy_model)
        
        self._log_offline_transitions()

        self.log_dict({
            "losses/total_loss": total_loss,
            "losses/bellman_loss": bellman_loss,
            "losses/cql_loss": cql_loss,
            "losses/entropy": entropy,
            "train/q_mean": all_q_values.mean(),
            "train/rules": float(len(self.rules))
        })

    def validation_step(self, batch, batch_idx):
        if not self.self_organized or self.fuzzy_model is None or self.target_fuzzy_model is None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and getattr(datamodule, "val_reader", None) is not None:
            val_batch = datamodule.val_reader.get_batch(batch, device="cpu")
            obs = val_batch["obs"]
            actions = val_batch["action"]
            rewards = val_batch["reward"]
            next_obs = val_batch["next_obs"]
            dones = val_batch["done"]
            
            with torch.no_grad():
                next_q = self.target_fuzzy_model(next_obs)
                next_v = torch.max(next_q, dim=1)[0]
                q_target = rewards + self.get_cfg("gamma", 0.99) * next_v * (1 - dones)
                
                all_q_values = self.fuzzy_model(obs)
                q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                bellman_loss = F.mse_loss(q_action, q_target)
                logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
                cql_alpha = getattr(self.fuzzy_model, "cql_alpha", self.get_cfg("cql_alpha", 1.0))
                cql_loss = (logsumexp_qvalues - q_action).mean()
                val_loss = bellman_loss + cql_alpha * cql_loss
                
            self.log("val/loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("val/bellman_loss", bellman_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
            self.log("val/cql_loss", cql_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
            return val_loss

    def configure_optimizers(self):
        # Dummy optimizer to satisfy Lightning until self_organize is called
        return optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-4)

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        if not self.self_organized or self.fuzzy_model is None:
            return torch.zeros(obs.shape[0], dtype=torch.long, device=self.device), \
                   torch.zeros(obs.shape[0], device=self.device), \
                   torch.zeros(obs.shape[0], device=self.device), \
                   torch.zeros(obs.shape[0], device=self.device)
        
        obs_cpu = obs.to("cpu")
        act, log_p, ent, val = self.fuzzy_model.get_action_and_value(obs_cpu)
        return act.to(self.device), log_p.to(self.device), ent.to(self.device), val.to(self.device)

    def get_value(self, obs, logic_obs=None):
        if not self.self_organized or self.fuzzy_model is None:
            return torch.zeros(obs.shape[0], device=self.device)
        obs_cpu = obs.to("cpu")
        _, _, _, val = self.fuzzy_model.get_action_and_value(obs_cpu)
        return val.to(self.device)
