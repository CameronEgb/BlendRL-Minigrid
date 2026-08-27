#!/usr/bin/env python3
"""
scripts/inspect_trained_models.py — Inspect learned NSFR clause weights and blending distributions across all checkpoints.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import torch as th
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CKPT_BASE = PROJECT_ROOT / "results" / "checkpoints" / "pyrenees" / "tune_pyrenees_3way"


def inspect_checkpoint(ckpt_path):
    try:
        ckpt = th.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": str(e)}

    sd = ckpt.get("state_dict", {})
    results = {
        "ckpt_name": ckpt_path.stem,
        "path": str(ckpt_path),
        "im_W": None,
        "blender_type": "unknown",
        "blender_weights": None,
    }

    # 1. Extract NSFR Clause Weights (im.W)
    for k in ["model.policy_modules.0.im.W", "model.actor.policy_modules.0.im.W"]:
        if k in sd:
            results["im_W"] = sd[k].detach().cpu().numpy().tolist()
            break

    # 2. Check Blender Type & Parameters
    if "model.actor.blender.actor.weight" in sd or "model.blender.actor.weight" in sd:
        results["blender_type"] = "neural_mlp"
        b_key = "model.actor.blender.actor.bias" if "model.actor.blender.actor.bias" in sd else "model.blender.actor.bias"
        if b_key in sd:
            bias = sd[b_key].detach().cpu().numpy()
            probs = np.exp(bias) / np.sum(np.exp(bias))
            results["blender_bias_probs"] = probs.tolist()
    else:
        results["blender_type"] = "logic_blender"

    return results


def main():
    print("=" * 80)
    print("      PYRENEES CHECKPOINT WEIGHTS & BLENDING INSPECTOR")
    print(f"      Searching: {CKPT_BASE}")
    print("=" * 80)

    if not CKPT_BASE.exists():
        print(f"Directory not found: {CKPT_BASE}")
        return

    ckpt_files = sorted(CKPT_BASE.glob("**/*.ckpt"))
    # Filter only top-level / best model files
    ckpt_files = [f for f in ckpt_files if not f.name.startswith("last") and not f.name.startswith("epoch=")]

    if not ckpt_files:
        print("No .ckpt files found in the directory.")
        return

    print(f"Found {len(ckpt_files)} checkpoint(s):\n")

    records = []
    for p in ckpt_files:
        info = inspect_checkpoint(p)
        if "error" in info:
            print(f"[{p.name}] Error loading: {info['error']}")
            continue

        name = p.stem
        w_str = str(info["im_W"]) if info["im_W"] is not None else "N/A"
        b_type = info["blender_type"]

        print(f"Model: {name}")
        print(f"  Path:         {p}")
        print(f"  Blender Mode: {b_type}")
        print(f"  Learned im.W: {w_str}")
        if "blender_bias_probs" in info:
            bp = info["blender_bias_probs"]
            print(f"  Blender Prior: Module 0 (Logic) = {bp[0]*100:.1f}%, Module 1 (Neural) = {bp[1]*100:.1f}%")
        print("-" * 80)

        records.append({
            "model_name": name,
            "blender_type": b_type,
            "im_W": w_str,
            "path": str(p),
        })

    out_csv = CKPT_BASE / "checkpoint_weights_summary.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nSaved summary CSV -> {out_csv}")


if __name__ == "__main__":
    main()
