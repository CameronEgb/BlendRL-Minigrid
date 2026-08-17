#!/usr/bin/env python3
"""
Evaluate clinical decision performance (AUC-ROC, F1, Admin Rate, Module Weights)
for all trained checkpoints from the MIMIC HP sweep.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "fyd_repo", "src"))
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

from src.methods.cql_agent import CQLAgent
from src.methods.cew_agent import CEWAgent
from src.methods.iql_agent import IQLAgent

def load_policy_agent(path, dev):
    for cls in [CQLAgent, CEWAgent, IQLAgent]:
        try:
            ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False)
            ag.to(dev)
            ag.eval()
            return ag
        except Exception as e:
            continue
    return None

def main():
    npz_path = Path("in/datasets/mimic/mimic_lazy_0_interventions_balanced.npz")
    if not npz_path.exists():
        print(f"Dataset {npz_path} not found!")
        return

    print("Loading MIMIC validation / full evaluation data...")
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    mask = data['mask']
    valid_mask = (mask.squeeze(-1) != -1)
    obs = X[:, :, :46][valid_mask]
    clin_acts = X[:, :, 47][valid_mask].astype(int)

    device = torch.device("cpu")
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

    ckpt_root = Path("results/checkpoints/mimic/mimic_hp_sweep")
    if not ckpt_root.exists():
        print(f"Checkpoints directory {ckpt_root} not found!")
        return

    results = []
    clin_admin_rate = (clin_acts == 1).mean() * 100.0
    results.append({
        "Model": "Clinician (Ground Truth)",
        "Admin Rate %": f"{clin_admin_rate:.2f}%",
        "AUC-ROC": "1.0000",
        "AUPRC": "1.0000",
        "Precision": "1.0000",
        "Recall": "1.0000",
        "F1": "1.0000",
        "Avg Blend β": "N/A"
    })

    for model_dir in sorted(ckpt_root.iterdir()):
        if not model_dir.is_dir():
            continue
        ckpts = list(model_dir.glob("**/best_model*.ckpt"))
        if not ckpts:
            continue
        ckpt_path = ckpts[0]
        ag = load_policy_agent(ckpt_path, device)
        if ag is None:
            print(f"Could not load agent from {ckpt_path}")
            continue

        with torch.no_grad():
            if hasattr(ag, "is_modular") and ag.is_modular:
                logic_obs = ag._prepare_logic_obs(obs_t) if hasattr(ag, "_prepare_logic_obs") else obs_t.unsqueeze(1).repeat(1, 2, 1)
                probs, weights = ag.model.actor(obs_t, logic_obs)
                admin_probs = probs[:, 1].cpu().numpy()
                pred_acts = torch.argmax(probs, dim=-1).cpu().numpy()
                avg_logic_w = weights[:, 0].mean().item() if weights is not None and weights.shape[1] > 0 else 0.0
                avg_neural_w = weights[:, 1].mean().item() if weights is not None and weights.shape[1] > 1 else 0.0
                weight_str = f"L:{avg_logic_w:.2f} / N:{avg_neural_w:.2f}"
            else:
                q_vals = ag.q_network(obs_t) if hasattr(ag, "q_network") else ag(obs_t)
                probs = torch.softmax(q_vals, dim=-1)
                admin_probs = probs[:, 1].cpu().numpy()
                pred_acts = torch.argmax(q_vals, dim=-1).cpu().numpy()
                weight_str = "Neural Only (1.0)"

        admin_rate = (pred_acts == 1).mean() * 100.0
        try:
            auc = roc_auc_score(clin_acts, admin_probs)
        except Exception:
            auc = 0.5
        try:
            auprc = average_precision_score(clin_acts, admin_probs)
        except Exception:
            auprc = 0.0
        prec = precision_score(clin_acts, pred_acts, zero_division=0)
        rec = recall_score(clin_acts, pred_acts, zero_division=0)
        f1 = f1_score(clin_acts, pred_acts, zero_division=0)

        results.append({
            "Model": model_dir.name,
            "Admin Rate %": f"{admin_rate:.2f}%",
            "AUC-ROC": f"{auc:.4f}",
            "AUPRC": f"{auprc:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1": f"{f1:.4f}",
            "Avg Blend β": weight_str
        })

    df = pd.DataFrame(results)
    print("\n" + "="*100)
    print("  CLINICAL DECISION METRICS ON SAVED CHECKPOINTS")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")

    out_csv = Path("results/plots/mimic/mimic_hp_sweep/clinical_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved clinical metrics to {out_csv}")

if __name__ == "__main__":
    main()
