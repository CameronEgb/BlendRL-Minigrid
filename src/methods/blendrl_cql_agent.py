import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict, Optional
from blendrl.agents.blender_agent import BlenderActorCritic
from src.methods.registry import register_agent
from src.methods.base_agent import OfflineAgentBase

@register_agent("blendrl_cql")
class BlendRLCQLAgent(OfflineAgentBase):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        self.lr = self.get_cfg("lr", 3e-4)
        
        from blendrl.env_vectorized import VectorizedNudgeBaseEnv
        # Use algorithm name for mode to handle custom rules
        algorithm = self.get_cfg("algorithm", self.get_cfg("name", cfg.env.name))
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
            self.get_cfg("rules", cfg.env.rules),
            self.get_cfg("actor_mode", "hybrid"),
            self.get_cfg("blender_mode", "neural"),
            self.get_cfg("blend_function", "softmax"),
            self.get_cfg("reasoner", cfg.env.reasoner),
            self.device,
            architecture=self.get_cfg("architecture", cfg.env.architecture),
            cfg=cfg.agent
        )
        
        self.target_model = BlenderActorCritic(
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
        self.target_model.load_state_dict(self.model.state_dict())


    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        
        # Handle interval-based dataset scaling
        datamodule = self.trainer.datamodule
        if hasattr(datamodule, "reader") and datamodule.reader is not None:
            epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
            
            # Re-run self-organization if needed
            if self.current_epoch % epochs_per_interval == 0:
                sample_size = min(len(datamodule.reader), 10000)
                if sample_size > 0:
                    batch = datamodule.reader.sample(sample_size)
                    organize_obs = batch["logic_obs"] if batch["logic_obs"] is not None else batch["obs"]
                    changed1 = self.model.self_organize_cew_modules(organize_obs)
                    changed2 = self.target_model.self_organize_cew_modules(organize_obs)
                    
                    if changed1 or changed2:
                        self.target_model.load_state_dict(self.model.state_dict())
                        # Re-initialize optimizer because parameters (IDs) changed
                        lr = self.get_cfg("lr", 3e-4)
                        self.opt = optim.Adam(self.model.parameters(), lr=lr)
                    elif not hasattr(self, "opt"):
                        # Ensure we have an optimizer if it wasn't created yet
                        self.opt = self.configure_optimizers()

    def training_step(self, batch, batch_idx):
        datamodule = self.trainer.datamodule
        batch_size = self.get_cfg("batch_size", 256)
        real_batch = datamodule.reader.sample(batch_size)
        
        obs = real_batch["obs"].to(self.device)
        if real_batch["logic_obs"] is not None:
            logic_obs = real_batch["logic_obs"].to(self.device)
        else:
            # Vectorized duplication: (B, 46) -> (B, 2, 46)
            logic_obs = obs.unsqueeze(1).repeat(1, 2, 1)
            
        actions = real_batch["action"].to(self.device)
        rewards = real_batch["reward"].to(self.device)
        next_obs = real_batch["next_obs"].to(self.device)
        
        if real_batch["next_logic_obs"] is not None:
            next_logic_obs = real_batch["next_logic_obs"].to(self.device)
        else:
            # Vectorized duplication: (B, 46) -> (B, 2, 46)
            next_logic_obs = next_obs.unsqueeze(1).repeat(1, 2, 1)
            
        dones = real_batch["done"].to(self.device)
        
        # Use self.opt directly to ensure we use the updated optimizer after re-org
        if not hasattr(self, "opt"):
            self.opt = self.optimizers()
        opt = self.opt
        
        with torch.no_grad():
            next_q = self.target_model.get_q_values(next_obs, next_logic_obs)
            next_v = torch.max(next_q, dim=1)[0]
            q_target = rewards + self.cfg.env.gamma * next_v * (1 - dones)
            
        all_q_values = self.model.get_q_values(obs, logic_obs)
        q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # CQL component: logsumexp(Q) - Q(s,a)
        logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
        cql_alpha = self.get_cfg("cql_alpha", 1.0)
        cql_loss = (logsumexp_qvalues - q_action).mean()
        
        bellman_loss = F.mse_loss(q_action, q_target)
        
        # Entropy bonus
        probs = torch.softmax(all_q_values, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
        
        total_loss = bellman_loss + cql_alpha * cql_loss - 0.01 * entropy
        
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()
        
        self._soft_update(self.model, self.target_model)
        
        # Logging
        epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
        current_interval = self.current_epoch // epochs_per_interval
        interval_size = self.cfg.total_timesteps // self.cfg.intervals_count
        current_transitions = interval_size * (current_interval + 1)

        self.log_dict({
            "losses/total_loss": total_loss,
            "losses/bellman_loss": bellman_loss,
            "losses/cql_loss": cql_loss,
            "losses/entropy": entropy,
            "train/q_mean": all_q_values.mean()
        })
        self.log("transitions", float(current_transitions), logger=False, prog_bar=True)


    def configure_optimizers(self):
        lr = self.get_cfg("lr", 3e-4)
        return optim.Adam(self.model.parameters(), lr=lr)

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        q = self.model.get_q_values(obs, logic_obs)
        if action is None:
            action = torch.argmax(q, dim=1)
        log_probs = torch.log_softmax(q, dim=1)
        ent = -(torch.softmax(q, dim=1) * log_probs).sum(dim=1)
        return action, log_probs.gather(1, action.unsqueeze(1)).squeeze(1), ent, torch.max(q, dim=1)[0]
