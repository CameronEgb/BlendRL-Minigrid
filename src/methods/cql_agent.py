import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from typing import Any, Dict, Optional

from src.methods.base_agent import OfflineAgentBase
from src.methods.registry import register_agent
from blendrl.agents.blender_agent import BlenderActorCritic


@register_agent(
    "cql",
    "cql_dnn",
    "blendrl_cql",
    "cql_blendrl_human_neural",
    "cql_blendrl_human_cew",
    "cql_blendrl_cew_only",
    "blendrl_cql_human_neural",
    "blendrl_cql_human_cew",
    "blendrl_cql_cew_only",
)
class CQLAgent(OfflineAgentBase):
    """Unified Conservative Q-Learning (CQL) Offline RL Agent.
    
    Supports pure neural architectures, logic-guided policies, neuro-fuzzy CEW modules,
    and hybrid BlendRL mixtures.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        self.lr = self.get_cfg("lr", 3e-4)
        
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
        self.logic_observation_space = dummy_logic.shape[1:]
        self.n_actions = self.env.n_actions if not callable(self.env.n_actions) else self.env.n_actions()

        # Check if modular/hybrid architecture is configured
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
        else:
            hidden_sizes = cfg.agent.get("hidden_sizes", [256, 256])
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

    def _prepare_logic_obs(self, obs, logic_obs=None):
        if logic_obs is not None:
            return logic_obs.to(self.device)
        if obs.ndim == 2:
            return obs.unsqueeze(1).repeat(1, 2, 1).to(self.device)
        return obs.to(self.device)

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        if self.is_modular:
            logic_obs = self._prepare_logic_obs(obs, logic_obs)
            return self.model(obs, logic_obs, action=action)
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
        if self.is_modular:
            logic_obs = self._prepare_logic_obs(obs, logic_obs)
            return self.model.get_value(obs, logic_obs)
        q_vals = self.q_network.get_q_values(obs)
        return q_vals.max(dim=-1)[0]

    def on_train_start(self):
        if hasattr(self.trainer.datamodule, "reader") and self.trainer.datamodule.reader is not None:
            self.trainer.datamodule.reader.device = self.device

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.is_modular:
            datamodule = self.trainer.datamodule
            if hasattr(datamodule, "reader") and datamodule.reader is not None:
                epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
                if self.current_epoch % epochs_per_interval == 0:
                    sample_size = min(len(datamodule.reader), 10000)
                    if sample_size > 0:
                        batch = datamodule.reader.sample(sample_size)
                        organize_obs = batch["logic_obs"] if batch["logic_obs"] is not None else batch["obs"]
                        changed1 = self.model.self_organize_cew_modules(organize_obs)
                        changed2 = self.target_model.self_organize_cew_modules(organize_obs)
                        if changed1 or changed2:
                            self.target_model.load_state_dict(self.model.state_dict())
                            lr = self.get_cfg("lr", 3e-4)
                            self.opt = optim.Adam(self.model.parameters(), lr=lr)

    def training_step(self, batch, batch_idx):
        datamodule = self.trainer.datamodule
        cfg = self.cfg
        batch_size = self.get_cfg("batch_size", 256)
        real_batch = datamodule.reader.sample(batch_size)
        
        obs = real_batch["obs"].to(self.device)
        actions = real_batch["action"].to(self.device)
        rewards = real_batch["reward"].to(self.device)
        next_obs = real_batch["next_obs"].to(self.device)
        dones = real_batch["done"].to(self.device)
        
        cql_alpha = self.get_cfg("cql_alpha", 1.0)
        gamma = cfg.env.gamma

        if self.is_modular:
            logic_obs = self._prepare_logic_obs(obs, real_batch.get("logic_obs"))
            next_logic_obs = self._prepare_logic_obs(next_obs, real_batch.get("next_logic_obs"))

            with torch.no_grad():
                next_q = self.target_model.get_q_values(next_obs, next_logic_obs)
                next_v = torch.max(next_q, dim=1)[0]
                q_target = rewards + gamma * next_v * (1 - dones)
                
            all_q_values = self.model.get_q_values(obs, logic_obs)
            q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            bellman_loss = F.mse_loss(q_action, q_target)
            logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
            cql_loss = (logsumexp_qvalues - q_action).mean()
            q_loss = bellman_loss + cql_alpha * cql_loss

            _, log_probs, entropy, blend_entropy, _ = self.model(obs, logic_obs, action=actions)
            with torch.no_grad():
                q_val_act = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            actor_loss = -(q_val_act * log_probs).mean() - 0.01 * entropy.mean()

            total_loss = q_loss + actor_loss
            
            opt = getattr(self, "opt", self.optimizers())
            if isinstance(opt, list):
                opt = opt[0]
            opt.zero_grad()
            self.manual_backward(total_loss)
            opt.step()
            
            soft_target_tau = self.get_cfg("soft_target_tau", 0.005)
            self._soft_update(self.model, self.target_model, tau=soft_target_tau)
        else:
            blend_entropy = None
            opt_q, opt_a = self.optimizers()
            with torch.no_grad():
                next_q = self.target_q_network(next_obs)
                next_v = torch.max(next_q, dim=1)[0]
                q_target = rewards + gamma * next_v * (1 - dones)
                
            all_q_values = self.q_network(obs)
            q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            bellman_loss = F.mse_loss(q_action, q_target)
            logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
            cql_loss = (logsumexp_qvalues - q_action).mean()
            q_loss = bellman_loss + cql_alpha * cql_loss
            
            opt_q.zero_grad()
            self.manual_backward(q_loss)
            opt_q.step()
            
            _, log_probs, entropy, _ = self.actor.get_action_and_value(obs, actions)
            with torch.no_grad():
                q_val_act = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            actor_loss = -(q_val_act * log_probs).mean() - 0.01 * entropy.mean()
            
            opt_a.zero_grad()
            self.manual_backward(actor_loss)
            opt_a.step()
            
            soft_target_tau = self.get_cfg("soft_target_tau", 0.005)
            self._soft_update(self.q_network, self.target_q_network, tau=soft_target_tau)

        self._log_offline_transitions()
        log_data = {
            "losses/total_loss": (q_loss + actor_loss).item() if isinstance(q_loss + actor_loss, torch.Tensor) else q_loss + actor_loss,
            "losses/q_loss": q_loss.item() if isinstance(q_loss, torch.Tensor) else q_loss,
            "losses/bellman_loss": bellman_loss.item() if isinstance(bellman_loss, torch.Tensor) else bellman_loss,
            "losses/cql_loss": cql_loss.item() if isinstance(cql_loss, torch.Tensor) else cql_loss,
            "losses/actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
        }
        if self.is_modular and isinstance(blend_entropy, torch.Tensor):
            log_data["losses/blend_entropy"] = blend_entropy.mean().item()
        self.log_dict(log_data)

    def validation_step(self, batch, batch_idx):
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and getattr(datamodule, "val_reader", None) is not None:
            val_batch = datamodule.val_reader.get_batch(batch, device=self.device)
            obs = val_batch["obs"]
            actions = val_batch["action"]
            rewards = val_batch["reward"]
            next_obs = val_batch["next_obs"]
            dones = val_batch["done"]
            cql_alpha = self.get_cfg("cql_alpha", 1.0)
            
            with torch.no_grad():
                if self.is_modular:
                    logic_obs = self._prepare_logic_obs(obs, val_batch.get("logic_obs"))
                    next_logic_obs = self._prepare_logic_obs(next_obs, val_batch.get("next_logic_obs"))
                    next_q = self.target_model.get_q_values(next_obs, next_logic_obs)
                    next_v = torch.max(next_q, dim=1)[0]
                    q_target = rewards + self.cfg.env.gamma * next_v * (1 - dones)
                    all_q_values = self.model.get_q_values(obs, logic_obs)
                else:
                    next_q = self.target_q_network(next_obs)
                    next_v = torch.max(next_q, dim=1)[0]
                    q_target = rewards + self.cfg.env.gamma * next_v * (1 - dones)
                    all_q_values = self.q_network(obs)
                    
                q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                bellman_loss = F.mse_loss(q_action, q_target)
                logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
                cql_loss = (logsumexp_qvalues - q_action).mean()
                val_loss = bellman_loss + cql_alpha * cql_loss
                
            self.log("val/loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("val/bellman_loss", bellman_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
            self.log("val/cql_loss", cql_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
            self.log("val/q_mean", all_q_values.mean(), prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
            return val_loss

    def configure_optimizers(self):
        if self.is_modular:
            lr = self.get_cfg("lr", 3e-4)
            return optim.Adam(self.model.parameters(), lr=lr)
        opt_q = optim.Adam(self.q_network.parameters(), lr=self.cfg.agent.lr)
        opt_a = optim.Adam(self.actor.parameters(), lr=self.cfg.agent.lr)
        return [opt_q, opt_a]
