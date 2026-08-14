#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root and src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.methods.cql_agent import CQLAgent
from src.methods.blendrl_cql_agent import BlendRLCQLAgent

def evaluate_mimic_policies(exp_id="mimic_test", group="mimic", dataset_path=None):
    if dataset_path is None:
        dataset_path = "in/datasets/mimic/mimic_lazy_0_interventions_balanced.npz"

    if not os.path.exists(dataset_path):
        print(f"Error: MIMIC dataset not found at {dataset_path}", flush=True)
        return

    print(f"=== Evaluating MIMIC Policy Metrics (Vectorized Batch Mode) ===", flush=True)
    print(f"Loading dataset: {dataset_path}", flush=True)
    data = np.load(dataset_path, allow_pickle=True)
    X = data['X']        # (N, 240, 49)
    mask = data['mask']  # (N, 240, 1)

    valid_mask = (mask.squeeze(-1) != -1) # (N, 240)
    all_obs = X[:, :, :46][valid_mask]    # (Total_Steps, 46)
    all_clin_acts = X[:, :, 47][valid_mask].astype(int) # (Total_Steps,)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    total_steps = len(all_clin_acts)
    print(f"Loaded {total_steps} valid transitions.", flush=True)

    ckpt_root = Path("results/checkpoints") / group / exp_id
    if not ckpt_root.exists():
        print(f"No checkpoints found at {ckpt_root}", flush=True)
        return

    # Discover agent checkpoints
    method_ckpts = {}
    for method_dir in ckpt_root.iterdir():
        if method_dir.is_dir():
            ckpts = list(method_dir.rglob("best_model*.ckpt"))
            if ckpts:
                method_ckpts[method_dir.name] = ckpts[0]

    results = []

    # 1. Clinician Baseline
    clin_admin_rate = (all_clin_acts == 1).mean() * 100.0
    results.append({
        "Method": "Clinician (Baseline)",
        "Accuracy %": 100.0,
        "Admin Rate %": float(clin_admin_rate),
        "AUC-ROC": 1.0000,
        "AUPRC": 1.0000,
        "Precision": 1.0000,
        "Recall": 1.0000,
        "F1 Score": 1.0000
    })

    # 2. Evaluate Policy Checkpoints in Batches
    batch_size = 10000
    for method_name, ckpt_path in sorted(method_ckpts.items()):
        print(f"Evaluating {method_name} from {ckpt_path}...", flush=True)
        agent = None
        try:
            agent = BlendRLCQLAgent.load_from_checkpoint(str(ckpt_path), map_location=device, weights_only=False)
        except Exception:
            try:
                agent = CQLAgent.load_from_checkpoint(str(ckpt_path), map_location=device, weights_only=False)
            except Exception as e:
                print(f"  Warning: Could not load checkpoint {ckpt_path}: {e}", flush=True)
                continue

        agent.to(device)
        agent.eval()

        all_admin_probs = []
        all_policy_acts = []

        with torch.no_grad():
            for b_start in range(0, total_steps, batch_size):
                b_end = min(b_start + batch_size, total_steps)
                obs_batch = torch.tensor(all_obs[b_start:b_end], dtype=torch.float32).to(device)

                if hasattr(agent, "actor") and hasattr(agent.actor, "get_action_probs"):
                    probs = agent.actor.get_action_probs(obs_batch)
                    policy_acts = torch.argmax(probs, dim=-1).cpu().numpy()
                    admin_probs = probs[:, 1].cpu().numpy()
                elif hasattr(agent, "model"):
                    q = agent.model.get_q_values(obs_batch, logic_state=None)
                    probs = torch.softmax(q, dim=-1)
                    policy_acts = torch.argmax(probs, dim=-1).cpu().numpy()
                    admin_probs = probs[:, 1].cpu().numpy()
                else:
                    continue

                all_admin_probs.extend(admin_probs)
                all_policy_acts.extend(policy_acts)

        all_admin_probs = np.array(all_admin_probs)
        all_policy_acts = np.array(all_policy_acts)

        matches = (all_policy_acts == all_clin_acts).sum()
        accuracy = (matches / total_steps) * 100.0
        admin_rate = (all_policy_acts == 1).mean() * 100.0

        tp = ((all_policy_acts == 1) & (all_clin_acts == 1)).sum()
        fp = ((all_policy_acts == 1) & (all_clin_acts == 0)).sum()
        fn = ((all_policy_acts == 0) & (all_clin_acts == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        auc_roc = roc_auc_score(all_clin_acts, all_admin_probs) if len(np.unique(all_clin_acts)) > 1 else 0.0
        auprc = average_precision_score(all_clin_acts, all_admin_probs) if len(np.unique(all_clin_acts)) > 1 else 0.0

        results.append({
            "Method": method_name,
            "Accuracy %": float(accuracy),
            "Admin Rate %": float(admin_rate),
            "AUC-ROC": float(auc_roc),
            "AUPRC": float(auprc),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1 Score": float(f1)
        })

    df = pd.DataFrame(results)
    out_dir = Path("results/plots") / group / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "methods_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved methods comparison CSV to: {csv_path}", flush=True)



if __name__ == "__main__":
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "mimic_test"
    evaluate_mimic_policies(exp_id)
