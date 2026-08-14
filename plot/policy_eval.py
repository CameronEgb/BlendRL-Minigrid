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

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.method_registry import get_style as get_method_style

        from src.methods.cql_agent import CQLAgent
        from src.methods.blendrl_cql_agent import BlendRLCQLAgent
        from src.methods.cew_agent import CEWAgent
        from src.methods.iql_agent import IQLAgent
        from src.methods.blendrl_iql_agent import BlendRLIQLAgent

        def load_policy_agent(path, dev):
            for cls in [BlendRLCQLAgent, CQLAgent, CEWAgent, BlendRLIQLAgent, IQLAgent]:
                try:
                    ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False)
                    ag.to(dev)
                    ag.eval()
                    return ag
                except Exception:
                    continue
            return None

        # Patient indexing for per-patient agreement
        num_patients = X.shape[0]
        outcomes = data['y'].squeeze() if 'y' in data else np.zeros(num_patients)
        patient_agreements = {}

        batch_size = 10000
        for method_name, ckpt_path in sorted(method_ckpts.items()):
            agent = load_policy_agent(ckpt_path, device)
            if agent is None:
                print(f"  Warning [policy_eval]: Could not load checkpoint {ckpt_path}")
                continue

            all_admin_probs = []
            all_policy_acts = []

            with torch.no_grad():
                for b_start in range(0, total_steps, batch_size):
                    b_end = min(b_start + batch_size, total_steps)
                    obs_batch = torch.tensor(all_obs[b_start:b_end], dtype=torch.float32).to(device)

                    if hasattr(agent, "get_action_and_value"):
                        act, log_p, ent, val = agent.get_action_and_value(obs_batch)
                        policy_acts = act.cpu().numpy() if isinstance(act, torch.Tensor) else np.array(act)
                    elif hasattr(agent, "actor") and hasattr(agent.actor, "get_action_probs"):
                        probs = agent.actor.get_action_probs(obs_batch)
                        policy_acts = torch.argmax(probs, dim=-1).cpu().numpy()
                    elif hasattr(agent, "actor"):
                        act, _, _, _ = agent.actor.get_action_and_value(obs_batch)
                        policy_acts = act.cpu().numpy()
                    elif hasattr(agent, "model"):
                        q = agent.model.get_q_values(obs_batch, logic_state=None)
                        policy_acts = torch.argmax(q, dim=-1).cpu().numpy()
                    else:
                        continue

                    if hasattr(agent, "get_action_probs"):
                        probs = agent.get_action_probs(obs_batch)
                        admin_probs = probs[:, 1].cpu().numpy() if probs.shape[-1] > 1 else probs.squeeze().cpu().numpy()
                    elif hasattr(agent, "actor") and hasattr(agent.actor, "get_action_probs"):
                        probs = agent.actor.get_action_probs(obs_batch)
                        admin_probs = probs[:, 1].cpu().numpy()
                    elif hasattr(agent, "q_network"):
                        q = agent.q_network(obs_batch)
                        probs = torch.softmax(q, dim=-1)
                        admin_probs = probs[:, 1].cpu().numpy()
                    elif hasattr(agent, "model"):
                        q = agent.model.get_q_values(obs_batch, logic_state=None)
                        probs = torch.softmax(q, dim=-1)
                        admin_probs = probs[:, 1].cpu().numpy()
                    else:
                        admin_probs = (policy_acts == 1).astype(float)

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

            # Compute per-patient agreement for Agreement vs Shock plot
            patient_agr = []
            curr_step = 0
            for i in range(num_patients):
                v_steps = (mask[i, :, 0] != -1).sum()
                if v_steps == 0:
                    patient_agr.append(np.nan)
                    continue
                p_policy = all_policy_acts[curr_step:curr_step + v_steps]
                p_clin = all_clin_acts[curr_step:curr_step + v_steps]
                curr_step += v_steps
                agr = (p_policy == p_clin).mean() * 100.0
                patient_agr.append(agr)
            patient_agreements[method_name] = np.array(patient_agr)

        df = pd.DataFrame(results)
        csv_path = output_dir / "methods_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # 2. Generate Agreement vs Shock Rate Plot for all methods
        if patient_agreements and len(outcomes) > 0:
            bins = np.linspace(0, 100, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2.0

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()

            first_counts = None

            for method_name, agr in patient_agreements.items():
                means, sems, counts = [], [], []
                for b_idx in range(10):
                    low, high = bins[b_idx], bins[b_idx + 1]
                    if b_idx == 9:
                        mask_bin = (agr >= low) & (agr <= high)
                    else:
                        mask_bin = (agr >= low) & (agr < high)
                    pts = outcomes[mask_bin]
                    counts.append(len(pts))
                    if len(pts) > 0:
                        means.append(float(np.mean(pts)) * 100.0)
                        sems.append(float(np.std(pts) / np.sqrt(len(pts))) * 100.0 if len(pts) > 1 else 0.0)
                    else:
                        means.append(np.nan)
                        sems.append(0.0)

                if first_counts is None:
                    first_counts = counts

                means_arr = np.array(means)
                sems_arr = np.array(sems)
                valid = ~np.isnan(means_arr)

                style = get_method_style(method_name)
                label = style["label"]
                color = style["color"]
                marker = style["marker"]

                ax1.plot(bin_centers[valid], means_arr[valid], marker=marker, color=color,
                         label=label, linewidth=2.5, markersize=7)
                ax1.fill_between(bin_centers[valid],
                                 means_arr[valid] - sems_arr[valid],
                                 means_arr[valid] + sems_arr[valid],
                                 color=color, alpha=0.12)

            if first_counts is not None:
                ax2.bar(bin_centers, first_counts, width=8, color='tab:blue', alpha=0.15,
                        label='Trajectory Count', zorder=1)
                ax2.set_ylabel("Patient Trajectory Count", fontsize=12, fontweight="bold", color="tab:blue")
                ax2.tick_params(axis='y', labelcolor="tab:blue")

            ax1.set_xlabel("Clinician – RL Policy Agreement (%)", fontsize=12, fontweight="bold")
            ax1.set_ylabel("True Septic Shock Rate (%)", fontsize=12, fontweight="bold")
            ax1.set_xticks(np.arange(0, 101, 10))
            ax1.grid(True, linestyle="--", alpha=0.5)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="best")

            ax1.set_title(f"Septic Shock Rate vs. Clinician Agreement ({exp_id})", fontsize=13, fontweight="bold")

            fig.tight_layout()
            plot_path = output_dir / "agreement_vs_shock.png"
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"  Saved: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Policy Evaluation Plotter")
    parser.add_argument("experiment_id", type=str, help="Experiment ID")
    args = parser.parse_args()
    PolicyEvalPlotter().run(args.experiment_id)
