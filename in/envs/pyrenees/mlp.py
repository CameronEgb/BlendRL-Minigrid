"""
Pyrenees Neural Policy MLP — for BlendRL hybrid and pure-neural baselines.

Input:  126-dim augmented state vector
        [0:123]  z-scored Pyrenees features
        [123]    last_was_ps  (one-hot alternation bit)
        [124]    last_was_we  (one-hot alternation bit)
        [125]    last_was_fwe (one-hot alternation bit)

Output: logits over 3 actions (PS=0, WE=1, FWE=2)
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(
        self,
        device,
        hidden_sizes=(256, 256),
        has_softmax=False,
        has_sigmoid=False,
        out_size=3,          # PS / WE / FWE
        as_dict=False,
        logic=False,
    ):
        super().__init__()
        self.device = device
        self.logic  = logic

        # 126-dim augmented input (123 features + 3 alternation bits)
        self.num_in_features = 126

        # ── Backbone ─────────────────────────────────────────────────────
        layers = []
        last_size = self.num_in_features
        for size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            layers.append(nn.ReLU())
            last_size = size
        self.network = nn.Sequential(*layers)

        # ── Heads ────────────────────────────────────────────────────────
        self.actor  = layer_init(nn.Linear(last_size, out_size), std=0.01)
        self.critic = layer_init(nn.Linear(last_size, 1),        std=1.0)

        # Optional activations
        self.softmax = nn.Softmax(dim=-1) if has_softmax else nn.Identity()
        self.sigmoid = nn.Sigmoid()       if has_sigmoid else nn.Identity()

        self.to(device)

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.network(self._flat(x))
        return self.softmax(self.actor(hidden))

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.network(self._flat(x))
        return torch.softmax(self.actor(hidden), dim=-1)

    def get_value(self, x: torch.Tensor, logic_state=None) -> torch.Tensor:
        return self.critic(self.network(self._flat(x)))

    def get_action_and_value(self, x: torch.Tensor, action=None):
        flat   = self._flat(x)
        hidden = self.network(flat)
        logits = self.actor(hidden)
        probs  = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def act(self, x: torch.Tensor, logic_state=None, epsilon: float = 0.0):
        probs = Categorical(probs=self.get_action_probs(x))
        action = probs.sample()
        return action, probs.log_prob(action)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.network(self._flat(x)))

    def _print(self) -> str:
        return (
            f"Pyrenees Neural Agent (MLP) — "
            f"In: {self.num_in_features}, Actions: 3 (PS/WE/FWE)"
        )
