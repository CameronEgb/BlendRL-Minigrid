"""
Sepsis Transformer Neural Policy & Q-Network for MIMIC-III / MIMIC-IV Clinical Tasks.

Architecture:
  - Feature-Tokenized Tabular Transformer (FT-Transformer).
  - 46 individual clinical variables are projected into high-dimensional embedding tokens.
  - Prepended learnable [CLS] classification token with learned feature position embeddings.
  - Multi-layer Transformer Encoder (Pre-LN, Multi-Head Self-Attention, GELU FeedForward).
  - Dual Pooling (CLS token + Soft Feature Attention Pooling).
  - Dueling Decomposition Heads: Value Stream V(s) and Advantage Stream A(s, a).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureAttentionPooling(nn.Module):
    """Soft attention pooling over feature tokens."""
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.Tanh(),
            nn.Linear(max(16, d_model // 2), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_tokens, d_model)
        scores = self.attn(x).squeeze(-1)  # (B, N_tokens)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, N_tokens, 1)
        return (x * weights).sum(dim=1)  # (B, d_model)


class SepsisTransformerPolicy(nn.Module):
    """
    Sepsis Transformer Policy & Q-Network for MIMIC Offline RL and Hybrid BlendRL.
    """
    def __init__(
        self,
        device=None,
        num_in_features: int = 46,
        out_size: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_dueling: bool = True,
        has_softmax: bool = False,
        has_sigmoid: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.num_in_features = num_in_features
        self.out_size = out_size
        self.d_model = d_model
        self.use_dueling = use_dueling

        # Feature Tokenizer: Each 1D scalar feature gets its own linear embedding weight
        self.feature_weights = nn.Parameter(torch.empty(num_in_features, d_model))
        self.feature_biases = nn.Parameter(torch.empty(num_in_features, d_model))
        nn.init.normal_(self.feature_weights, std=0.02)
        nn.init.zeros_(self.feature_biases)

        # Learnable CLS token & Position embeddings (1 CLS token + 46 feature tokens = 47)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_in_features + 1, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.input_layer_norm = nn.LayerNorm(d_model)

        # Transformer Encoder Stack (Pre-LN for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = FeatureAttentionPooling(d_model)

        # Latent representation: [CLS] token (d_model) + Attention Pooled features (d_model) = 2 * d_model
        latent_dim = d_model * 2

        # ── Dueling Value Head V(s) ───────────────────────────────────
        self.value_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            layer_init(nn.Linear(latent_dim, d_model)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            layer_init(nn.Linear(d_model, 1), std=1.0)
        )

        # ── Dueling Advantage Head A(s, a) ───────────────────────────
        self.advantage_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            layer_init(nn.Linear(latent_dim, d_model)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            layer_init(nn.Linear(d_model, out_size), std=0.01)
        )

        # Backward-compatible references
        self.critic = self.value_head
        self.actor = self.advantage_head

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
        B = flat.shape[0]

        # Tokenize each scalar feature: x_i * w_i + b_i -> (B, num_in_features, d_model)
        tokens = flat.unsqueeze(-1) * self.feature_weights.unsqueeze(0) + self.feature_biases.unsqueeze(0)

        # Prepend CLS token: (B, 1 + num_in_features, d_model)
        cls_expanded = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_expanded, tokens], dim=1)
        seq = seq + self.pos_embedding
        seq = self.input_layer_norm(seq)

        # Self-Attention Encoder
        out = self.transformer_encoder(seq)  # (B, 1 + num_in_features, d_model)

        cls_out = out[:, 0]  # (B, d_model)
        feat_tokens = out[:, 1:]  # (B, num_in_features, d_model)
        pooled_out = self.attn_pool(feat_tokens)  # (B, d_model)

        # Combined representation
        latent = torch.cat([cls_out, pooled_out], dim=-1)  # (B, 2 * d_model)
        return latent

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.extract_features(x)
        if self.use_dueling:
            val = self.value_head(latent)
            adv = self.advantage_head(latent)
            return val + (adv - adv.mean(dim=-1, keepdim=True))
        return self.advantage_head(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return self.softmax(q)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return torch.softmax(q, dim=-1)

    def get_value(self, x: torch.Tensor, logic_state=None) -> torch.Tensor:
        latent = self.extract_features(x)
        return self.value_head(latent)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        latent = self.extract_features(x)
        v = self.value_head(latent)
        a = self.advantage_head(latent)
        q = v + (a - a.mean(dim=-1, keepdim=True)) if self.use_dueling else a
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
        return f"MIMIC Sepsis Transformer Policy - In: {self.num_in_features}, D_Model: {self.d_model}, Trainable Params: {param_count:,}"


class CrossAttentionSepsisPolicy(nn.Module):
    """
    Cross-Attention Neuro-Symbolic Hybrid Policy & Q-Network for MIMIC.
    
    Architecture:
      - Queries: Patient feature tokens (46 clinical features + [CLS] token)
      - Keys & Values: First-order logic clauses and predicate valuation embeddings
      - Multi-Head Cross-Attention fuses clinical vitals directly with symbolic rules
      - Dual Pooling + Dueling Value & Advantage decomposition heads
    """
    def __init__(
        self,
        device=None,
        num_in_features: int = 46,
        num_rules: int = 8,
        out_size: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_dueling: bool = True,
        has_softmax: bool = False,
        has_sigmoid: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.num_in_features = num_in_features
        self.num_rules = num_rules
        self.out_size = out_size
        self.d_model = d_model
        self.use_dueling = use_dueling

        # Feature Tokenizer: Each 1D scalar feature gets its own linear embedding weight
        self.feature_weights = nn.Parameter(torch.empty(num_in_features, d_model))
        self.feature_biases = nn.Parameter(torch.empty(num_in_features, d_model))
        nn.init.normal_(self.feature_weights, std=0.02)
        nn.init.zeros_(self.feature_biases)

        # Symbolic Rule Memory Embeddings (Keys & Values)
        self.rule_embeddings = nn.Parameter(torch.empty(num_rules, d_model))
        nn.init.normal_(self.rule_embeddings, std=0.02)
        self.rule_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # [CLS] token & Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_in_features + 1, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.input_layer_norm = nn.LayerNorm(d_model)

        # Cross-Attention Layer: Queries=Patient Features, Keys/Values=Symbolic Rules
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = FeatureAttentionPooling(d_model)

        latent_dim = d_model * 2

        # ── Dueling Value Head V(s) ───────────────────────────────────
        self.value_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            layer_init(nn.Linear(latent_dim, d_model)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            layer_init(nn.Linear(d_model, 1), std=1.0)
        )

        # ── Dueling Advantage Head A(s, a) ───────────────────────────
        self.advantage_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            layer_init(nn.Linear(latent_dim, d_model)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            layer_init(nn.Linear(d_model, out_size), std=0.01)
        )

        self.critic = self.value_head
        self.actor = self.advantage_head

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

    def _get_rule_tokens(self, x: torch.Tensor, logic_obs: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        rule_tokens = self.rule_embeddings.unsqueeze(0).expand(B, -1, -1)
        
        if logic_obs is not None:
            if logic_obs.ndim == 3 and logic_obs.shape[-1] >= 46:
                val_lactate = torch.sigmoid((logic_obs[:, 0, 10] - 1.0) / 0.5)
                val_bp = torch.sigmoid((-1.0 - logic_obs[:, 0, 15]) / 0.5)
                val_creat = torch.sigmoid((logic_obs[:, 0, 12] - 1.0) / 0.5)
                val_bili = torch.sigmoid((logic_obs[:, 0, 13] - 1.0) / 0.5)
                val_plat = torch.sigmoid((-1.0 - logic_obs[:, 0, 11]) / 0.5)
                
                r0 = val_bp * val_lactate
                r1 = val_bp
                r2 = val_lactate
                r3 = val_creat
                r4 = val_bili
                r5 = val_plat
                r6 = (1.0 - val_bp) * (1.0 - val_lactate) * (1.0 - val_creat)
                r7 = torch.ones_like(r0)
                
                rule_weights = torch.stack([r0, r1, r2, r3, r4, r5, r6, r7], dim=-1).unsqueeze(-1)
                rule_tokens = rule_tokens * rule_weights
                
        return self.rule_proj(rule_tokens)

    def extract_features(self, x: torch.Tensor, logic_obs: torch.Tensor = None) -> torch.Tensor:
        flat = self._flat(x)
        B = flat.shape[0]

        tokens = flat.unsqueeze(-1) * self.feature_weights.unsqueeze(0) + self.feature_biases.unsqueeze(0)
        cls_expanded = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_expanded, tokens], dim=1) + self.pos_embedding
        seq = self.input_layer_norm(seq)

        rule_kv = self._get_rule_tokens(flat, logic_obs)
        cross_out, _ = self.cross_attn(query=seq, key=rule_kv, value=rule_kv)
        seq = self.cross_norm(seq + cross_out)

        out = self.transformer_encoder(seq)
        cls_out = out[:, 0]
        feat_tokens = out[:, 1:]
        pooled_out = self.attn_pool(feat_tokens)

        return torch.cat([cls_out, pooled_out], dim=-1)

    def get_q_values(self, x: torch.Tensor, logic_obs: torch.Tensor = None) -> torch.Tensor:
        latent = self.extract_features(x, logic_obs)
        if self.use_dueling:
            val = self.value_head(latent)
            adv = self.advantage_head(latent)
            return val + (adv - adv.mean(dim=-1, keepdim=True))
        return self.advantage_head(latent)

    def forward(self, x: torch.Tensor, logic_obs: torch.Tensor = None) -> torch.Tensor:
        q = self.get_q_values(x, logic_obs)
        return self.softmax(q)

    def get_action_probs(self, x: torch.Tensor, logic_obs: torch.Tensor = None) -> torch.Tensor:
        q = self.get_q_values(x, logic_obs)
        return torch.softmax(q, dim=-1)

    def get_value(self, x: torch.Tensor, logic_state=None) -> torch.Tensor:
        latent = self.extract_features(x, logic_state)
        return self.value_head(latent)

    def get_action_and_value(self, x: torch.Tensor, logic_obs=None, action=None):
        latent = self.extract_features(x, logic_obs)
        v = self.value_head(latent)
        a = self.advantage_head(latent)
        q = v + (a - a.mean(dim=-1, keepdim=True)) if self.use_dueling else a
        probs = torch.softmax(q, dim=-1)
        dist = Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), v

    def act(self, x: torch.Tensor, logic_state=None, epsilon: float = 0.0):
        probs = self.get_action_probs(x, logic_state)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def _print(self) -> str:
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"MIMIC Cross-Attention Sepsis Policy - In: {self.num_in_features}, Rules: {self.num_rules}, Trainable Params: {param_count:,}"


# Alias for backward compatibility
MLP = SepsisTransformerPolicy
