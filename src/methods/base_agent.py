"""
Base agent classes — shared interface and utilities for all RL agents.

Hierarchy:
    BaseAgent (ABC)
    ├── OnlineAgentBase   — rollout buffers, env stepping, GAE, dataset writing
    └── OfflineAgentBase  — interval limit calc, reader.sample(), target networks
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.optim as optim
import lightning as L
from omegaconf import DictConfig
import numpy as np


class BaseAgent(L.LightningModule, ABC):
    """Abstract base class for all RL agents in the BlendRL framework.
    
    Provides:
        - Unified config traversal (get_cfg)
        - Soft target network updates (_soft_update)  
        - Standard interface contract via abstract methods
        - Common environment initialization helpers
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

    # ──────────────────────────────────────────────
    # Config Utilities
    # ──────────────────────────────────────────────

    def get_cfg(self, key, default=None):
        """Unified config traversal — searches agent, env, then top-level config.
        
        Handles Hydra's nested DictConfig structures, including cases where
        agent config contains a nested 'agent' key from inheritance.
        """
        cfg = self.cfg
        
        # Search in agent config (with recursive nesting support)
        if hasattr(cfg, "agent"):
            if key in cfg.agent:
                return cfg.agent[key]
            # Handle double-nested agent config from Hydra inheritance
            if "agent" in cfg.agent and isinstance(cfg.agent.agent, (dict, DictConfig)):
                if key in cfg.agent.agent:
                    return cfg.agent.agent[key]
        
        # Search in env config
        if hasattr(cfg, "env") and key in cfg.env:
            return cfg.env[key]
        
        # Search in top-level config
        if key in cfg:
            return cfg[key]
        
        return default

    # ──────────────────────────────────────────────
    # Network Utilities
    # ──────────────────────────────────────────────

    def _soft_update(self, model, target_model):
        """Polyak averaging for target network updates."""
        tau = self.get_cfg("soft_target_tau", 0.005)
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # ──────────────────────────────────────────────
    # Environment Helpers
    # ──────────────────────────────────────────────

    def _init_env(self, n_envs=None):
        """Initialize the vectorized environment and extract observation/action spaces.
        
        Args:
            n_envs: Number of parallel environments. Defaults to cfg value for online,
                    1 for offline (evaluation only).
        
        Returns:
            Tuple of (dummy_logic_obs, dummy_neural_obs) from env.reset().
        """
        from blendrl.env_vectorized import VectorizedNudgeBaseEnv
        
        if n_envs is None:
            n_envs = self.get_cfg("num_envs", 4)
        
        algorithm = self.get_cfg("algorithm", self.get_cfg("name", self.cfg.env.name))
        
        self.env = VectorizedNudgeBaseEnv.from_name(
            self.cfg.env.name,
            n_envs=n_envs,
            mode=algorithm,
            seed=self.get_cfg("seed", self.cfg.seed)
        )
        
        dummy_logic, dummy_neural = self.env.reset()
        self.observation_space = dummy_neural.shape[1:]
        self.logic_observation_space = dummy_logic.shape[1:]
        self.n_actions = self.env.n_actions if not callable(self.env.n_actions) else self.env.n_actions()
        
        return dummy_logic, dummy_neural

    # ──────────────────────────────────────────────
    # Abstract Interface (enforced contract)
    # ──────────────────────────────────────────────

    @abstractmethod
    def get_action_and_value(self, obs, logic_obs=None, action=None):
        """Compute action, log probability, entropy, and value for given observations.
        
        Returns:
            Tuple of (action, logprob, entropy, value) — or with blend_entropy for hybrid agents.
        """
        ...

    @abstractmethod
    def get_value(self, obs, logic_obs=None):
        """Compute value estimate for given observations."""
        ...


class OfflineAgentBase(BaseAgent):
    """Base class for all offline RL agents (IQL, CQL, CEW, and their BlendRL variants).
    
    Provides:
        - Device transfer for train and validation readers
        - Interval-based dataset limit management (on_train_epoch_start)
        - Common offline training epoch tracking
    """

    def on_train_start(self):
        """Preload datasets to agent device on training start."""
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None:
            if hasattr(datamodule, "reader") and datamodule.reader is not None:
                datamodule.reader.device = self.device
            if hasattr(datamodule, "val_reader") and datamodule.val_reader is not None:
                datamodule.val_reader.device = self.device

    def on_train_epoch_start(self):
        """Set dataset limit based on current training interval.
        
        Implements the progressive data exposure schedule defined by
        intervals_count and epochs_per_interval (when intervals_count > 1).
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "reader") and datamodule.reader is not None:
            intervals_count = self.cfg.get("intervals_count", 1)
            if intervals_count > 1:
                epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
                current_interval = self.current_epoch // epochs_per_interval
                interval_size = self.cfg.total_timesteps // intervals_count
                current_limit = interval_size * (current_interval + 1)
                datamodule.reader.set_limit(min(current_limit, len(datamodule.reader)))
            else:
                datamodule.reader.set_limit(len(datamodule.reader))

    def _log_offline_transitions(self):
        """Calculate and log the current transition count for offline training."""
        cfg = self.cfg
        intervals_count = cfg.get("intervals_count", 1)
        if intervals_count > 1:
            epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
            current_interval = self.current_epoch // epochs_per_interval
            interval_size = cfg.total_timesteps // intervals_count
            current_transitions = interval_size * (current_interval + 1)
        else:
            current_transitions = cfg.total_timesteps if hasattr(cfg, "total_timesteps") and isinstance(cfg.total_timesteps, (int, float)) else len(self.trainer.datamodule.reader)
        self.log("transitions", float(current_transitions), logger=False, prog_bar=True)
        return current_transitions
