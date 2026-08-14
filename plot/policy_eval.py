#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.metrics import roc_auc_score, average_precision_score

from plot.base import BasePlotter

class PolicyEvalPlotter(BasePlotter):
    def __init__(self):
        super().__init__("policy_eval")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        
        env_name = cfg.get("env", {}).get("name", "mimic") if isinstance(cfg.get("env"), dict) else str(cfg.get("env", "mimic"))
        if env_name != "mimic":
            # Policy alignment evaluation is designed for MIMIC clinical decisions
            return

        env_ds = cfg.get("env", {}).get("dataset_name", "mimic_lazy_0_interventions_balanced.npz") if isinstance(cfg.get("env"), dict) else "mimic_lazy_0_interventions_balanced.npz"
        npz_candidate = Path("in/datasets/mimic") / env_ds
        if not npz_candidate.exists():
            # Check if mode.dataset_path points to an npz or dataset directory with matching npz
            mode_path = cfg.get("mode", {}).get("dataset_path", "")
            if mode_path:
                cand = Path(mode_path).with_suffix(".npz")
                if cand.exists():
                    npz_candidate = cand

        if not npz_candidate.exists():
            print(f"Notice [policy_eval]: NPZ dataset '{npz_candidate}' not found, skipping policy eval metrics.")
            return

        print(f"=== Running Policy Evaluation Module for '{exp_id}' ===")
        data = np.load(npz_candidate, allow_pickle=True)
        X = data['X']        # (N, 240, 49)
        mask = data['mask']  # (N, 240, 1)

        valid_mask = (mask.squeeze(-1) != -1)
        all_obs = X[:, :, :46][valid_mask]
        all_clin_acts = X[:, :, 47][valid_mask].astype(int)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        total_steps = len(all_clin_acts)

        ckpt_root = Path("results/checkpoints") / group / exp_id
        if not ckpt_root.exists():
            print(f"Notice [policy_eval]: No checkpoints found at {ckpt_root}")
            return

        # Discover agent checkpoints
        method_ckpts = {}
        for method_dir in ckpt_root.iterdir():
            if method_dir.is_dir():
                ckpts = list(method_dir.rglob("best_model*.ckpt"))
                if ckpts:
                    method_ckpts[method_dir.name] = ckpts[0]

        if not method_ckpts:
            print(f"Notice [policy_eval]: No policy checkpoints found in {ckpt_root}")
            return

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

        # Add project root and src to sys.path if not present
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

        from src.methods.cql_agent import CQLAgent
        from src.methods.blendrl_cql_agent import BlendRLCQLAgent

        batch_size = 10000
        for method_name, ckpt_path in sorted(method_ckpts.items()):
            agent = None
            try:
                agent = BlendRLCQLAgent.load_from_checkpoint(str(ckpt_path), map_location=device, weights_only=False)
            except Exception:
                try:
                    agent = CQLAgent.load_from_checkpoint(str(ckpt_path), map_location=device, weights_only=False)
                except Exception as e:
                    print(f"  Warning [policy_eval]: Could not load checkpoint {ckpt_path}: {e}")
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
        csv_path = output_dir / "methods_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Policy Evaluation Plotter")
    parser.add_argument("experiment_id", type=str, help="Experiment ID")
    args = parser.parse_args()

    plotter = PolicyEvalPlotter()
    plotter.run(args.experiment_id)
