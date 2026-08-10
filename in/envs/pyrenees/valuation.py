"""
Pyrenees valuation functions for BlendRL NSFR reasoning.

Each function maps a batch of logic state vectors to soft probabilities in [0, 1].
The logic state has shape (B, n_objects, 126), where the augmented 126-dim vector is:
    [0:123]  - original z-scored features from pyrenees_clean.npz
    [123]    - last_was_ps  (1.0 if last action was PS, else 0.0)
    [124]    - last_was_we  (1.0 if last action was WE, else 0.0)
    [125]    - last_was_fwe (1.0 if last action was FWE, else 0.0)

Feature indices (z-scored):
    72  - pctCorrect        (overall % correct)
    76  - pctCorrectKC      (KC-level % correct)
    80  - pctCorrectSession (session-level % correct)

Competency thresholds (from KMeans k=3 on per-student mean profiles):
    Low  | Med  boundary: pctCorrect z-score = -1.74  (LOW_MED_THRESH)
    Med  | High boundary: pctCorrect z-score = +0.02  (MED_HIGH_THRESH)

A secondary KC-level check tightens the boundary to reduce misclassification.
"""

import torch as th
from nsfr.utils.common import bool_to_probs

# ─── KMeans-derived competency thresholds (z-score space) ────────────────────
# Primary axis: pctCorrect (index 72)
LOW_MED_THRESH = -1.74    # below this → low competency
MED_HIGH_THRESH = 0.02    # above this → high competency

# Secondary axis: pctCorrectKC (index 76) — used to sharpen the boundaries
KC_LOW_THRESH  = -0.33    # pctCorrectKC below this pulls toward low
KC_HIGH_THRESH = -0.02    # pctCorrectKC above this pulls toward high

# Feature indices in the augmented 126-dim state
IDX_PCT_CORRECT     = 72   # pctCorrect (z-scored)
IDX_PCT_CORRECT_KC  = 76   # pctCorrectKC (z-scored)
IDX_PCT_CORRECT_SES = 80   # pctCorrectSession (z-scored)
IDX_LAST_PS         = 123  # augmented: 1.0 if last action was PS
IDX_LAST_WE         = 124  # augmented: 1.0 if last action was WE
IDX_LAST_FWE        = 125  # augmented: 1.0 if last action was FWE

# ─── Competency tier valuations ───────────────────────────────────────────────

def low_competency(agent: th.Tensor) -> th.Tensor:
    """
    Low competency: pctCorrect < LOW_MED_THRESH
    OR (pctCorrect < MED_HIGH_THRESH AND pctCorrectKC < KC_LOW_THRESH)
    """
    pc = agent[..., IDX_PCT_CORRECT]
    kc = agent[..., IDX_PCT_CORRECT_KC]
    is_low = (pc < LOW_MED_THRESH) | ((pc < MED_HIGH_THRESH) & (kc < KC_LOW_THRESH))
    return bool_to_probs(is_low)


def med_competency(agent: th.Tensor) -> th.Tensor:
    """
    Medium competency: LOW_MED_THRESH <= pctCorrect < MED_HIGH_THRESH
    (and not pulled into low by KC signal)
    """
    pc = agent[..., IDX_PCT_CORRECT]
    kc = agent[..., IDX_PCT_CORRECT_KC]
    is_low = (pc < LOW_MED_THRESH) | ((pc < MED_HIGH_THRESH) & (kc < KC_LOW_THRESH))
    is_high = (pc >= MED_HIGH_THRESH) & (kc >= KC_HIGH_THRESH)
    is_med = ~is_low & ~is_high
    return bool_to_probs(is_med)


def high_competency(agent: th.Tensor) -> th.Tensor:
    """
    High competency: pctCorrect >= MED_HIGH_THRESH
    AND pctCorrectKC >= KC_HIGH_THRESH
    """
    pc = agent[..., IDX_PCT_CORRECT]
    kc = agent[..., IDX_PCT_CORRECT_KC]
    is_high = (pc >= MED_HIGH_THRESH) & (kc >= KC_HIGH_THRESH)
    return bool_to_probs(is_high)


# ─── Alternation tracking valuations ─────────────────────────────────────────

def last_was_ps(agent: th.Tensor) -> th.Tensor:
    """True if the previous problem assigned was a Problem Solving (PS) type."""
    return bool_to_probs(agent[..., IDX_LAST_PS] > 0.5)


def last_was_we(agent: th.Tensor) -> th.Tensor:
    """True if the previous problem assigned was a Worked Example (WE) type."""
    return bool_to_probs(agent[..., IDX_LAST_WE] > 0.5)


def last_was_fwe(agent: th.Tensor) -> th.Tensor:
    """True if the previous problem assigned was a Faded Worked Example (FWE) type."""
    return bool_to_probs(agent[..., IDX_LAST_FWE] > 0.5)


# ─── Fallback ─────────────────────────────────────────────────────────────────

def true(env_obj: th.Tensor) -> th.Tensor:
    """Always-true predicate, used as a fallback catch-all rule."""
    return bool_to_probs(th.ones_like(env_obj[..., 0], dtype=th.bool))
