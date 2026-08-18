"""
Pyrenees Neural Policy MLP — for BlendRL hybrid and pure-neural baselines.

Input:  126-dim augmented state vector (auto-padded if 123-dim from dataset)
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
        flat = x.float().reshape(x.shape[0], -1)
        if flat.shape[-1] < self.num_in_features:
            pad = torch.zeros(
                (flat.shape[0], self.num_in_features - flat.shape[-1]),
                dtype=flat.dtype, device=flat.device
            )
            flat = torch.cat([flat, pad], dim=-1)
        return flat

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

class ResBlock(nn.Module):
    """Residual block with LayerNorm, GELU, and Dropout for Pyrenees representations."""
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = layer_init(nn.Linear(dim, dim))
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = layer_init(nn.Linear(dim, dim))
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.act2(out)
        out = self.drop2(out)
        return residual + out


class StandardMLP(nn.Module):
    """Standard Feedforward Multi-Layer Perceptron for Pyrenees."""
    def __init__(
        self,
        device=None,
        hidden_sizes=(256, 256),
        has_softmax=False,
        has_sigmoid=False,
        out_size=3,
        logic=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.logic = logic
        self.num_in_features = 126
        self.out_size = out_size

        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [256, 256]
        else:
            hidden_sizes = list(hidden_sizes)

        layers = []
        last_dim = self.num_in_features
        for h in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_dim, h)))
            layers.append(nn.ReLU())
            last_dim = h
        self.network = nn.Sequential(*layers)
        self.actor = layer_init(nn.Linear(last_dim, out_size), std=0.01)
        self.critic = layer_init(nn.Linear(last_dim, 1), std=1.0)
        self.softmax = nn.Softmax(dim=-1) if has_softmax else nn.Identity()

        if self.device is not None:
            self.to(self.device)

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.float().reshape(x.shape[0], -1)
        if flat.shape[-1] < self.num_in_features:
            pad = torch.zeros((flat.shape[0], self.num_in_features - flat.shape[-1]), dtype=flat.dtype, device=flat.device)
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[-1] > self.num_in_features:
            flat = flat[:, :self.num_in_features]
        return flat

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        flat = self._flat(x)
        hidden = self.network(flat)
        return self.actor(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return self.softmax(q)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return torch.softmax(q, dim=-1)

    def get_value(self, x: torch.Tensor, logic_state=None) -> torch.Tensor:
        flat = self._flat(x)
        hidden = self.network(flat)
        return self.critic(hidden)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        flat = self._flat(x)
        hidden = self.network(flat)
        q = self.actor(hidden)
        v = self.critic(hidden)
        probs = torch.softmax(q, dim=-1)
        dist = Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), v

    def act(self, x: torch.Tensor, logic_state=None, epsilon: float = 0.0):
        probs = self.get_action_probs(x)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def _print(self) -> str:
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Pyrenees Standard MLP Agent — In: {self.num_in_features}, Trainable Params: {param_count:,}"


class DuelingResNetMLP(nn.Module):
    """
    Scaled Dueling ResNet Neural Policy / Q-Network for Pyrenees ITS.
    
    Features:
      - Scaled deep representations with LayerNorm, GELU activations, and Residual skip connections.
      - Dueling Q-Value decomposition: Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a'))).
      - Decouples overall student mastery state V(s) from pedagogical action advantage A(s, a).
    """
    def __init__(
        self,
        device=None,
        hidden_sizes=(512, 512, 256, 128),
        has_softmax=False,
        has_sigmoid=False,
        out_size=3,
        as_dict=False,
        logic=False,
        use_dueling=True,
        dropout=0.05,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.logic = logic
        self.use_dueling = use_dueling
        self.num_in_features = 126

        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [512, 512, 256, 128]
        else:
            hidden_sizes = list(hidden_sizes)

        # ── Feature Extractor Backbone ──────────────────────────────
        embed_dim = hidden_sizes[0]
        self.input_proj = nn.Sequential(
            layer_init(nn.Linear(self.num_in_features, embed_dim)),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        layers = []
        curr_dim = embed_dim
        for next_dim in hidden_sizes[1:]:
            if next_dim == curr_dim:
                layers.append(ResBlock(curr_dim, dropout=dropout))
            else:
                layers.append(nn.Sequential(
                    nn.LayerNorm(curr_dim),
                    layer_init(nn.Linear(curr_dim, next_dim)),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ))
                curr_dim = next_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        final_dim = curr_dim

        # ── Dueling Value Head V(s) ───────────────────────────────────
        self.value_head = nn.Sequential(
            nn.LayerNorm(final_dim),
            layer_init(nn.Linear(final_dim, max(64, final_dim // 2))),
            nn.GELU(),
            layer_init(nn.Linear(max(64, final_dim // 2), 1), std=1.0)
        )

        # ── Dueling Advantage Head A(s, a) ───────────────────────────
        self.advantage_head = nn.Sequential(
            nn.LayerNorm(final_dim),
            layer_init(nn.Linear(final_dim, max(64, final_dim // 2))),
            nn.GELU(),
            layer_init(nn.Linear(max(64, final_dim // 2), out_size), std=0.01)
        )

        self.critic = self.value_head
        self.actor = self.advantage_head
        self.network = nn.Sequential(self.input_proj, self.backbone)

        self.softmax = nn.Softmax(dim=-1) if has_softmax else nn.Identity()
        self.sigmoid = nn.Sigmoid() if has_sigmoid else nn.Identity()

        if self.device is not None:
            self.to(self.device)

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.float().reshape(x.shape[0], -1)
        if flat.shape[-1] < self.num_in_features:
            pad = torch.zeros(
                (flat.shape[0], self.num_in_features - flat.shape[-1]),
                dtype=flat.dtype, device=flat.device
            )
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[-1] > self.num_in_features:
            flat = flat[:, :self.num_in_features]
        return flat

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        flat = self._flat(x)
        emb = self.input_proj(flat)
        return self.backbone(emb)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        if self.use_dueling:
            val = self.value_head(features)
            adv = self.advantage_head(features)
            return val + (adv - adv.mean(dim=-1, keepdim=True))
        return self.advantage_head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return self.softmax(q)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return torch.softmax(q, dim=-1)

    def get_value(self, x: torch.Tensor, logic_state=None) -> torch.Tensor:
        features = self.extract_features(x)
        return self.value_head(features)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        features = self.extract_features(x)
        if self.use_dueling:
            val = self.value_head(features)
            adv = self.advantage_head(features)
            q = val + (adv - adv.mean(dim=-1, keepdim=True))
        else:
            q = self.advantage_head(features)
            val = self.value_head(features)
        
        probs = torch.softmax(q, dim=-1)
        dist = Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), val

    def act(self, x: torch.Tensor, logic_state=None, epsilon: float = 0.0):
        probs = self.get_action_probs(x)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def _print(self) -> str:
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Pyrenees Dueling ResNet Agent — In: {self.num_in_features}, Trainable Params: {param_count:,}"
