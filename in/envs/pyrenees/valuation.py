"""
Pyrenees valuation functions for BlendRL NSFR reasoning.

Supports:
  1. Multi-dimensional Gaussian Mixture Model (GMM) posterior probabilities:
     - Uses 6 key performance, fluency, help-seeking, and pacing features:
       pctCorrect, pctCorrectKC, pctCorrectSession, nStepSinceLastWrong, nTotalHintSession, avgTimeOnStep.
     - Dynamically loads per-problem GMM parameters from in/datasets/pyrenees/per_problem/{problem_id}/gmm_scaler.npz.
     - Returns continuous soft probabilities P(Low|Med|High | s) in [0.01, 0.99] directly via Bayes' Rule.
  2. Alternation predicates:
     - last_was_ps, last_was_we, last_was_fwe.

Supports both Problem-Level (130-dim, 133-augmented) and Exercise Step-Level (123-dim, 126-augmented) inputs.
"""

import os
from pathlib import Path
import numpy as np
import torch as th
from nsfr.utils.common import bool_to_probs

_GMM_PARAMS_CACHE = {}
_CURRENT_PROBLEM_TYPE = None


def set_problem_type(problem_type: str):
    """Set the active problem type for GMM valuation loading."""
    global _CURRENT_PROBLEM_TYPE
    _CURRENT_PROBLEM_TYPE = problem_type


def _get_active_problem_type():
    global _CURRENT_PROBLEM_TYPE
    if _CURRENT_PROBLEM_TYPE is not None:
        return _CURRENT_PROBLEM_TYPE
    return os.environ.get("PYRENEES_PROBLEM_TYPE", "problem")


def _load_gmm_params_for_problem(problem_type: str = None):
    global _GMM_PARAMS_CACHE
    if problem_type is None:
        problem_type = _get_active_problem_type()

    if problem_type in _GMM_PARAMS_CACHE:
        return _GMM_PARAMS_CACHE[problem_type]

    # Candidate paths for this problem
    candidates = []
    env_gmm_path = os.environ.get("PYRENEES_GMM_PATH", "")
    if env_gmm_path and os.path.exists(env_gmm_path):
        candidates.append(env_gmm_path)

    # 1. Per-problem dedicated GMM scaler
    candidates.append(f"in/datasets/pyrenees/per_problem/{problem_type}/gmm_scaler.npz")
    candidates.append(f"in/datasets/pyrenees/per_problem/{problem_type}/scaler.npz")
    
    # 2. Global fallbacks
    candidates.extend([
        "in/datasets/pyrenees/per_problem/problem/gmm_scaler.npz",
        "in/datasets/pyrenees/pyrenees_gmm_scaler.npz",
        "in/datasets/pyrenees/pyrenees_scaler.npz",
    ])

    for p in candidates:
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
                    params = {
                        "feature_indices": feat_indices,
                        "means": means,
                        "precisions": precisions,
                        "log_dets": log_dets,
                        "log_weights": log_weights,
                    }
                    _GMM_PARAMS_CACHE[problem_type] = params
                    return params
            except Exception as e:
                print(f"[valuation.py] Warning: Failed to load GMM parameters from {p}: {e}")

    # Fallback to default GMM if problem specific not found
    print(f"[valuation.py] Warning: Could not locate GMM parameters for {problem_type}. Using uniform fallback.")
    return None


def _ensure_augmented(agent: th.Tensor) -> th.Tensor:
    """Ensure agent tensor has 3 alternation slots at the end."""
    last_dim = agent.shape[-1]
    # If unaugmented (123 or 130 features), append 3 zero slots
    if last_dim in [123, 130] or last_dim < 126:
        pad = th.zeros((*agent.shape[:-1], 3), dtype=agent.dtype, device=agent.device)
        return th.cat([agent, pad], dim=-1)
    return agent


def _compute_gmm_posteriors(agent: th.Tensor) -> th.Tensor:
    """
    Computes exact PyTorch log-posterior probabilities for Low, Med, High competency.
    Returns tensor of shape (*agent.shape[:-1], 3) with probabilities in [0.01, 0.99].
    """
    agent = _ensure_augmented(agent)
    device = agent.device
    dtype = agent.dtype

    gmm_p = _load_gmm_params_for_problem()
    if gmm_p is None:
        # Graceful fallback: equal 1/3 prior
        shape = (*agent.shape[:-1], 3)
        return th.full(shape, 1.0 / 3.0, dtype=dtype, device=device)

    feat_idx = th.tensor(gmm_p["feature_indices"], dtype=th.long, device=device)
    means = th.tensor(gmm_p["means"], dtype=dtype, device=device)             # (3, d)
    precisions = th.tensor(gmm_p["precisions"], dtype=dtype, device=device)   # (3, d, d)
    log_dets = th.tensor(gmm_p["log_dets"], dtype=dtype, device=device)       # (3,)
    log_weights = th.tensor(gmm_p["log_weights"], dtype=dtype, device=device) # (3,)

    x = agent[..., feat_idx]  # (*batch, d)
    d = x.shape[-1]
    const = 0.5 * d * np.log(2.0 * np.pi)

    log_probs = []
    for k in range(3):
        diff = x - means[k]  # (*batch, d)
        maha = th.sum(diff * th.matmul(diff, precisions[k]), dim=-1)  # (*batch,)
        log_p = log_weights[k] - 0.5 * log_dets[k] - 0.5 * maha - const
        log_probs.append(log_p)

    log_probs_tensor = th.stack(log_probs, dim=-1)    # (*batch, 3)
    posteriors = th.softmax(log_probs_tensor, dim=-1) # (*batch, 3)

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
    agent = _ensure_augmented(agent)
    return bool_to_probs(agent[..., -3] > 0.5)


def last_was_we(agent: th.Tensor) -> th.Tensor:
    agent = _ensure_augmented(agent)
    return bool_to_probs(agent[..., -2] > 0.5)


def last_was_fwe(agent: th.Tensor) -> th.Tensor:
    agent = _ensure_augmented(agent)
    return bool_to_probs(agent[..., -1] > 0.5)


# ─── Fallback ─────────────────────────────────────────────────────────────────

def true(env_obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.ones_like(env_obj[..., 0], dtype=th.bool))
