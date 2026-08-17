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
    "cql_dueling_resnet",
    "cql_transformer",
    "cql_sepsis_transformer",
    "blendrl_cql",
    "cql_blendrl_human_neural",
    "cql_blendrl_human_rigid",
    "cql_blendrl_human_continuous",
    "cql_blendrl_human_dueling_resnet",
    "cql_blendrl_human_dueling_resnet_rigid",
    "cql_blendrl_human_dueling_resnet_continuous",
    "cql_blendrl_human_transformer",
    "cql_blendrl_cross_attention",
    "cql_blendrl_human_cew",
    "cql_blendrl_cew_only",
    "blendrl_cql_human_neural",
    "blendrl_cql_human_rigid",
    "blendrl_cql_human_continuous",
    "blendrl_cql_human_dueling_resnet",
    "blendrl_cql_human_dueling_resnet_rigid",
    "blendrl_cql_human_dueling_resnet_continuous",
    "blendrl_cql_human_transformer",
    "blendrl_cql_cross_attention",
    "blendrl_cql_human_cew",
    "blendrl_cql_cew_only",
)
class CQLAgent(OfflineAgentBase):
    """Unified Conservative Q-Learning (CQL) Offline RL Agent.
    
    Supports pure neural architectures (MLP, Dueling ResNet, Sepsis Transformer),
    logic-guided policies, neuro-fuzzy CEW modules, and hybrid BlendRL mixtures.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        self.lr = self.get_cfg("lr", 3e-4)
        
        self._init_env(n_envs=1)
        algorithm = self.get_cfg("algorithm", self.get_cfg("name", cfg.env.name))

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
            
            architecture = self.get_cfg("architecture", cfg.env.architecture)
            if architecture in ["transformer", "sepsis_transformer", "cross_attention", "dueling_resnet", "resnet"]:
                from src.utils import get_neural_agent
                self.q_network = get_neural_agent(cfg.env.name, self.n_actions, self.device, arch_name=architecture, hidden_sizes=hidden_sizes)
                self.target_q_network = get_neural_agent(cfg.env.name, self.n_actions, self.device, arch_name=architecture, hidden_sizes=hidden_sizes)
            elif architecture in ["mlp", "dnn", "standard_mlp"]:
                from src.utils import MLPQNetwork
                self.q_network = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features, hidden_sizes=hidden_sizes)
                self.target_q_network = MLPQNetwork(n_actions=self.n_actions, num_in_features=num_in_features, hidden_sizes=hidden_sizes)
            else:
                from src.utils import QNetwork
                self.q_network = QNetwork(n_actions=self.n_actions)
                self.target_q_network = QNetwork(n_actions=self.n_actions)
                
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        target_admin_rate = self.get_cfg("target_admin_rate", None)
        self.target_admin_rate = float(target_admin_rate) if target_admin_rate is not None else None
        init_pos_w = 5.0
        try:
            val = float(self.get_cfg("pos_action_weight", 5.0))
            init_pos_w = val
        except (ValueError, TypeError):
            pass
        self.register_buffer("log_pos_weight", torch.tensor(math.log(max(init_pos_w, 1.0))))
        self.pos_weight_lr = float(self.get_cfg("pos_weight_lr", 0.005))

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
        q_vals = self.q_network.get_q_values(obs) if hasattr(self.q_network, "get_q_values") else self.q_network(obs)
        probs = torch.softmax(q_vals, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = torch.argmax(q_vals, dim=-1)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
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
        if hasattr(self.trainer.datamodule, "val_reader") and self.trainer.datamodule.val_reader is not None:
            self.trainer.datamodule.val_reader.device = self.device

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
            raise RuntimeError("CQLAgent requires an active offline dataset reader or batched dictionary.")
        
        obs = real_batch["obs"].to(self.device, non_blocking=True)
        actions = real_batch["action"].to(self.device, non_blocking=True)
        rewards = real_batch["reward"].to(self.device, non_blocking=True)
        reward_scale = float(self.get_cfg("reward_scale", 1.0))
        if reward_scale != 1.0:
            rewards = rewards * reward_scale

        next_obs = real_batch["next_obs"].to(self.device, non_blocking=True)
        dones = real_batch["done"].to(self.device, non_blocking=True)
        
        cql_alpha = self.get_cfg("cql_alpha", 1.0)
        gamma = cfg.env.gamma

        pos_action_weight_cfg = self.get_cfg("pos_action_weight", 1.0)
        is_adaptive = (str(pos_action_weight_cfg).lower() == "adaptive") or (self.target_admin_rate is not None)

        if is_adaptive:
            pos_weight = torch.exp(self.log_pos_weight).clamp(min=1.0, max=50.0)
            sample_weights = torch.where(actions == 1, pos_weight, torch.ones_like(actions, dtype=torch.float32))
        elif str(pos_action_weight_cfg).lower() == "auto":
            n_pos = (actions == 1).sum().float()
            n_neg = (actions == 0).sum().float()
            pos_weight = torch.clamp(n_neg / torch.clamp(n_pos, min=1.0), min=1.0, max=100.0)
            sample_weights = torch.where(actions == 1, pos_weight, torch.ones_like(actions, dtype=torch.float32))
        elif float(pos_action_weight_cfg) > 1.0:
            pos_weight = float(pos_action_weight_cfg)
            sample_weights = torch.where(actions == 1, torch.full_like(actions, pos_weight, dtype=torch.float32), torch.ones_like(actions, dtype=torch.float32))
        else:
            pos_weight = 1.0
            sample_weights = torch.ones_like(actions, dtype=torch.float32)

        weight_norm = sample_weights.mean().clamp(min=1e-8)

        bellman_loss_fn = str(self.get_cfg("bellman_loss", "smooth_l1")).lower()

        if self.is_modular:
            logic_obs = self._prepare_logic_obs(obs, real_batch.get("logic_obs"))
            next_logic_obs = self._prepare_logic_obs(next_obs, real_batch.get("next_logic_obs"))

            with torch.no_grad():
                online_next_q = self.model.get_q_values(next_obs, next_logic_obs)
                best_next_action = torch.argmax(online_next_q, dim=1, keepdim=True)
                next_v = self.target_model.get_q_values(next_obs, next_logic_obs).gather(1, best_next_action).squeeze(1)
                q_target = rewards + gamma * next_v * (1 - dones)
                
            all_q_values = self.model.get_q_values(obs, logic_obs)
            q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            if bellman_loss_fn in ["smooth_l1", "huber"]:
                bellman_element = F.smooth_l1_loss(q_action, q_target, beta=1.0, reduction="none")
            else:
                bellman_element = (q_action - q_target) ** 2
            bellman_loss = (sample_weights * bellman_element).mean() / weight_norm
            cql_diff = torch.logsumexp(all_q_values, dim=1) - q_action
            cql_loss = (sample_weights * cql_diff).mean() / weight_norm
            q_loss = bellman_loss + cql_alpha * cql_loss

            probs, weights = self.model.actor(obs, logic_obs)
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum(dim=1)
            blend_entropy = -(weights * torch.log(weights + 1e-12)).sum(dim=1) if weights is not None else None
            ent_coef = self.get_cfg("ent_coef", 0.01)
            blend_ent_coef = self.get_cfg("blend_ent_coef", 0.01)
            blend_entropy_loss = blend_entropy.mean() if isinstance(blend_entropy, torch.Tensor) else 0.0
            
            weighted_actor_obj = (sample_weights * (probs * all_q_values.detach()).sum(dim=1)).mean() / weight_norm
            actor_loss = -weighted_actor_obj - ent_coef * entropy.mean() - blend_ent_coef * blend_entropy_loss

            total_loss = q_loss + actor_loss
            
            opt = getattr(self, "opt", self.optimizers())
            if isinstance(opt, list):
                opt = opt[0]
            opt.zero_grad()
            self.manual_backward(total_loss)
            opt.step()
            
            soft_target_tau = self.get_cfg("soft_target_tau", 0.005)
            self._soft_update(self.model, self.target_model, tau=soft_target_tau)

            if is_adaptive:
                with torch.no_grad():
                    pred_admin = (torch.argmax(probs, dim=-1) == 1).float().mean()
                    tar = self.target_admin_rate if self.target_admin_rate is not None else 0.04
                    err = (tar - pred_admin).clamp(min=-1.0, max=1.0)
                    self.log_pos_weight += self.pos_weight_lr * err
                    self.log_pos_weight.clamp_(min=math.log(1.0), max=math.log(50.0))
        else:
            blend_entropy = None
            opt = getattr(self, "opt", self.optimizers())
            if isinstance(opt, list):
                opt = opt[0]
            with torch.no_grad():
                online_next_q = self.q_network(next_obs)
                best_next_action = torch.argmax(online_next_q, dim=1, keepdim=True)
                next_v = self.target_q_network(next_obs).gather(1, best_next_action).squeeze(1)
                q_target = rewards + gamma * next_v * (1 - dones)
                
            all_q_values = self.q_network(obs)
            q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            if bellman_loss_fn in ["smooth_l1", "huber"]:
                bellman_element = F.smooth_l1_loss(q_action, q_target, beta=1.0, reduction="none")
            else:
                bellman_element = (q_action - q_target) ** 2
            bellman_loss = (sample_weights * bellman_element).mean() / weight_norm
            cql_diff = torch.logsumexp(all_q_values, dim=1) - q_action
            cql_loss = (sample_weights * cql_diff).mean() / weight_norm
            q_loss = bellman_loss + cql_alpha * cql_loss
            
            opt.zero_grad()
            self.manual_backward(q_loss)
            opt.step()
            
            soft_target_tau = self.get_cfg("soft_target_tau", 0.005)
            self._soft_update(self.q_network, self.target_q_network, tau=soft_target_tau)

            if is_adaptive:
                with torch.no_grad():
                    pred_admin = (torch.argmax(all_q_values, dim=-1) == 1).float().mean()
                    tar = self.target_admin_rate if self.target_admin_rate is not None else 0.04
                    err = (tar - pred_admin).clamp(min=-1.0, max=1.0)
                    self.log_pos_weight += self.pos_weight_lr * err
                    self.log_pos_weight.clamp_(min=math.log(1.0), max=math.log(50.0))

            actor_loss = 0.0

        self._log_offline_transitions()
        log_data = {
            "losses/total_loss": (q_loss + actor_loss).item() if isinstance(q_loss + actor_loss, torch.Tensor) else q_loss + actor_loss,
            "losses/q_loss": q_loss.item() if isinstance(q_loss, torch.Tensor) else q_loss,
            "losses/bellman_loss": bellman_loss.item() if isinstance(bellman_loss, torch.Tensor) else bellman_loss,
            "losses/cql_loss": cql_loss.item() if isinstance(cql_loss, torch.Tensor) else cql_loss,
            "losses/actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
            "losses/pos_action_weight": float(pos_weight.item()) if isinstance(pos_weight, torch.Tensor) else float(pos_weight),
        }
        if self.is_modular and isinstance(blend_entropy, torch.Tensor):
            log_data["losses/blend_entropy"] = blend_entropy.mean().item()
        self.log_dict(log_data)

    def on_validation_epoch_start(self):
        self._val_step_losses = []

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
        reward_scale = float(self.get_cfg("reward_scale", 1.0))
        if reward_scale != 1.0:
            rewards = rewards * reward_scale

        next_obs = val_batch["next_obs"].to(self.device, non_blocking=True)
        dones = val_batch["done"].to(self.device, non_blocking=True)
        cql_alpha = self.get_cfg("cql_alpha", 1.0)
        bellman_loss_fn = str(self.get_cfg("bellman_loss", "smooth_l1")).lower()
            
        with torch.no_grad():
            if self.is_modular:
                logic_obs = self._prepare_logic_obs(obs, val_batch.get("logic_obs"))
                next_logic_obs = self._prepare_logic_obs(next_obs, val_batch.get("next_logic_obs"))
                online_next_q = self.model.get_q_values(next_obs, next_logic_obs)
                best_next_action = torch.argmax(online_next_q, dim=1, keepdim=True)
                next_v = self.target_model.get_q_values(next_obs, next_logic_obs).gather(1, best_next_action).squeeze(1)
                q_target = rewards + self.cfg.env.gamma * next_v * (1 - dones)
                all_q_values = self.model.get_q_values(obs, logic_obs)
                probs, _ = self.model.actor(obs, logic_obs)
                pred_acts = torch.argmax(probs, dim=-1)
            else:
                online_next_q = self.q_network(next_obs)
                best_next_action = torch.argmax(online_next_q, dim=1, keepdim=True)
                next_v = self.target_q_network(next_obs).gather(1, best_next_action).squeeze(1)
                q_target = rewards + self.cfg.env.gamma * next_v * (1 - dones)
                all_q_values = self.q_network(obs)
                pred_acts = torch.argmax(all_q_values, dim=-1)
                
            q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            if bellman_loss_fn in ["smooth_l1", "huber"]:
                bellman_loss = F.smooth_l1_loss(q_action, q_target, beta=1.0)
            else:
                bellman_loss = F.mse_loss(q_action, q_target)
            logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
            cql_loss = (logsumexp_qvalues - q_action).mean()
            val_loss = bellman_loss + cql_alpha * cql_loss
            admin_rate = (pred_acts == 1).float().mean()
            
        self.log("val/loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/bellman_loss", bellman_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/cql_loss", cql_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/q_mean", all_q_values.mean(), prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/admin_rate", admin_rate, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        if hasattr(self, "_val_step_losses"):
            self._val_step_losses.append(val_loss.detach())
        return val_loss

    def on_validation_epoch_end(self):
        if hasattr(self, "_val_step_losses") and len(self._val_step_losses) > 0:
            losses = torch.stack(self._val_step_losses)
            mean_loss = losses.mean()
            std_loss = losses.std() if len(losses) > 1 else torch.tensor(0.0, device=losses.device)
            robust_loss = mean_loss + 3.0 * std_loss
            self.log("val/robust_loss", robust_loss, prog_bar=True, sync_dist=True)
            self.log("val/loss_std", std_loss, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        weight_decay = self.get_cfg("weight_decay", 0.0)
        lr = self.get_cfg("lr", 3e-4)
        if self.is_modular:
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        return optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=weight_decay)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.is_modular:
            cew_states = []
            for i, m in enumerate(self.model.policy_modules):
                if i < len(self.model.module_types) and self.model.module_types[i] == "cew":
                    cew_states.append({
                        "index": i,
                        "antecedents": getattr(m, "antecedents", None),
                        "rules": getattr(m, "rules", None),
                        "n_inputs": getattr(m, "n_inputs", None),
                        "n_outputs": getattr(m, "n_outputs", None),
                    })
            if cew_states:
                checkpoint["cew_modules_state"] = cew_states

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        sd = checkpoint.get("state_dict", {})
        if self.is_modular:
            from src.methods.cew_utils import MultiFLC
            for i, m_type in enumerate(self.model.module_types):
                if m_type == "cew":
                    prefix = f"model.policy_modules.{i}."
                    if f"{prefix}flcs.0.links" in sd:
                        new_m = MultiFLC.from_state_dict_shapes(prefix, sd, self.n_actions)
                        self.model.policy_modules[i] = new_m
                        self.model.actor.policy_modules[i] = new_m
                        
                        target_prefix = f"target_model.policy_modules.{i}."
                        new_target = MultiFLC.from_state_dict_shapes(target_prefix, sd, self.n_actions)
                        self.target_model.policy_modules[i] = new_target
                        self.target_model.actor.policy_modules[i] = new_target
