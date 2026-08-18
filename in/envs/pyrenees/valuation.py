"""
Pyrenees valuation functions for BlendRL NSFR reasoning.

Supports both:
  1. Multi-dimensional Gaussian Mixture Model (GMM) posterior probabilities:
     - Uses 6 key performance, fluency, help-seeking, and pacing features:
       pctCorrect (72), pctCorrectKC (76), pctCorrectSession (80),
       nStepSinceLastWrong (84), nTotalHintSession (43), avgTimeOnStep (23).
     - Returns continuous soft probabilities P(Low|Med|High | s) in [0, 1] directly via Bayes' Rule.
  2. Legacy KMeans thresholding fallback if GMM parameters are unavailable.

Logic state shape: (B, n_objects, D) where D is 126 (or 123 if unpadded):
    [0:123]  - z-scored performance features
    [123]    - last_was_ps  (1.0 if last action was PS, else 0.0)
    [124]    - last_was_we  (1.0 if last action was WE, else 0.0)
    [125]    - last_was_fwe (1.0 if last action was FWE, else 0.0)
"""

import os
import numpy as np
import torch as th
from nsfr.utils.common import bool_to_probs

# ─── Feature & Slot Indices ──────────────────────────────────────────────────
IDX_PCT_CORRECT     = 72   # pctCorrect (z-scored)
IDX_PCT_CORRECT_KC  = 76   # pctCorrectKC (z-scored)
IDX_PCT_CORRECT_SES = 80   # pctCorrectSession (z-scored)
IDX_LAST_PS         = 123  # augmented: 1.0 if last action was PS
IDX_LAST_WE         = 124  # augmented: 1.0 if last action was WE
IDX_LAST_FWE        = 125  # augmented: 1.0 if last action was FWE


# ─── Load GMM Parameters (if available) ──────────────────────────────────────
_GMM_PARAMS = None
_SCALER_PATHS = [
    "in/datasets/pyrenees/pyrenees_gmm_scaler.npz",
    "in/datasets/pyrenees/pyrenees_scaler.npz",
]

def _load_or_fit_gmm_params():
    global _GMM_PARAMS
    if _GMM_PARAMS is not None:
        return _GMM_PARAMS

    for p in _SCALER_PATHS:
        if os.path.exists(p):
            try:
                data = np.load(p, allow_pickle=True)
                if "precisions" in data or "gmm_precisions" in data:
                    prefix = "" if "precisions" in data else "gmm_"
                    feat_indices = data.get(f"{prefix}feature_indices", np.array([72, 76, 80, 84, 43, 23]))
                    means = data[f"{prefix}means"]
                    precisions = data[f"{prefix}precisions"]
                    log_dets = data[f"{prefix}log_dets"]
                    log_weights = data[f"{prefix}log_weights"]
                    _GMM_PARAMS = {
                        "feature_indices": feat_indices,
                        "means": means,
                        "precisions": precisions,
                        "log_dets": log_dets,
                        "log_weights": log_weights,
                    }
                    print(f"[valuation.py] Successfully loaded GMM competency parameters from {p}")
                    return _GMM_PARAMS
            except Exception as e:
                print(f"[valuation.py] Warning: Failed to load GMM parameters from {p}: {e}")

    ds_path = "in/datasets/pyrenees/pyrenees_clean.npz"
    if os.path.exists(ds_path):
        try:
            print("[valuation.py] Auto-generating GMM competency parameters from dataset...")
            import importlib.util
            script_path = "scripts/fit_gmm_competency.py"
            if os.path.exists(script_path):
                spec = importlib.util.spec_from_file_location("fit_gmm", script_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.main()
                return _load_or_fit_gmm_params()
        except Exception as e:
            print(f"[valuation.py] Warning: Could not auto-generate GMM parameters: {e}")

    return None

_load_or_fit_gmm_params()


def _ensure_126(agent: th.Tensor) -> th.Tensor:
    """Pad tensor to 126 dimensions if coming from a 123-dim unaugmented batch."""
    if agent.shape[-1] < 126:
        pad_size = 126 - agent.shape[-1]
        pad = th.zeros((*agent.shape[:-1], pad_size), dtype=agent.dtype, device=agent.device)
        return th.cat([agent, pad], dim=-1)
    return agent


def _compute_gmm_posteriors(agent: th.Tensor) -> th.Tensor:
    """
    Computes exact PyTorch log-posterior probabilities for Low, Med, High competency.
    Returns tensor of shape (*agent.shape[:-1], 3) with probabilities in [0.01, 0.99].
    """
    agent = _ensure_126(agent)
    device = agent.device
    dtype  = agent.dtype

    gmm_p = _load_or_fit_gmm_params()
    if gmm_p is None:
        raise FileNotFoundError(
            "Pyrenees GMM competency parameters not found. "
            "Please run 'python scripts/fit_gmm_competency.py' to generate 'in/datasets/pyrenees/pyrenees_gmm_scaler.npz' "
            "before evaluating logic rules on Pyrenees."
        )

    feat_idx    = th.tensor(gmm_p["feature_indices"], dtype=th.long, device=device)
    means       = th.tensor(gmm_p["means"], dtype=dtype, device=device)             # (3, d)
    precisions  = th.tensor(gmm_p["precisions"], dtype=dtype, device=device)        # (3, d, d)
    log_dets    = th.tensor(gmm_p["log_dets"], dtype=dtype, device=device)          # (3,)
    log_weights = th.tensor(gmm_p["log_weights"], dtype=dtype, device=device)       # (3,)

    x = agent[..., feat_idx]  # (*batch, d)
    d = x.shape[-1]
    const = 0.5 * d * np.log(2.0 * np.pi)

    log_probs = []
    for k in range(3):
        diff = x - means[k]  # (*batch, d)
        # Mahalanobis distance = diff^T @ precisions[k] @ diff
        maha = th.sum(diff * th.matmul(diff, precisions[k]), dim=-1)  # (*batch,)
        log_p = log_weights[k] - 0.5 * log_dets[k] - 0.5 * maha - const
        log_probs.append(log_p)

    log_probs_tensor = th.stack(log_probs, dim=-1)  # (*batch, 3)
    posteriors = th.softmax(log_probs_tensor, dim=-1) # (*batch, 3)

    # Clamp to [0.01, 0.99] for NSFR numerical stability
    return th.clamp(posteriors, min=0.01, max=0.99)


# ─── Competency Tier Valuations ───────────────────────────────────────────────

def low_competency(agent: th.Tensor) -> th.Tensor:
    posteriors = _compute_gmm_posteriors(agent)
    return posteriors[..., 0]


def med_competency(agent: th.Tensor) -> th.Tensor:
    posteriors = _compute_gmm_posteriors(agent)
    return posteriors[..., 1]


def high_competency(agent: th.Tensor) -> th.Tensor:
    posteriors = _compute_gmm_posteriors(agent)
    return posteriors[..., 2]


# ─── Alternation Tracking Valuations ─────────────────────────────────────────

def last_was_ps(agent: th.Tensor) -> th.Tensor:
    agent = _ensure_126(agent)
    return bool_to_probs(agent[..., IDX_LAST_PS] > 0.5)


def last_was_we(agent: th.Tensor) -> th.Tensor:
    agent = _ensure_126(agent)
    return bool_to_probs(agent[..., IDX_LAST_WE] > 0.5)


def last_was_fwe(agent: th.Tensor) -> th.Tensor:
    agent = _ensure_126(agent)
    return bool_to_probs(agent[..., IDX_LAST_FWE] > 0.5)


# ─── Fallback ─────────────────────────────────────────────────────────────────

def true(env_obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.ones_like(env_obj[..., 0], dtype=th.bool))
