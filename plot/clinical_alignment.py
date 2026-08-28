#!/usr/bin/env python3
"""
plot/clinical_alignment.py — MIMIC Clinical Policy Alignment & Septic Shock Plotter.

Evaluates trained RL policies against clinician treatment decisions and patient
septic shock outcomes in ICU time-series trajectories.

Outputs:
  - Clinical action alignment: precision, recall, F1, windowed F1, AUC-ROC, AUPRC
  - Septic shock outcome analysis across clinician agreement deciles
  - Visual figures: clinical_agreement.png, clinician_agreement_vs_shock.png,
    auc_comparison.png, pr_curves.png
  - Markdown summary: clinical_alignment_report.md
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure project root and src are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from plot.base import BasePlotter, clean_label, get_canonical_method_name, get_method_aliases
from src.method_registry import get_style as get_method_style


class ClinicalAlignmentPlotter(BasePlotter):
    def __init__(self):
        super().__init__("clinical_alignment")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        self._run_mimic_eval(exp_id, cfg, group, clean_exp, output_dir)

    def _discover_checkpoints(self, exp_id: str, group: str, clean_exp: str):
        ckpt_root = Path("results/checkpoints") / group / clean_exp
        if not ckpt_root.exists():
            ckpt_root = Path("results/checkpoints") / clean_exp
        if not ckpt_root.exists():
            return {}

        exp_cfg = self.get_experiment_config(exp_id)
        active_aliases = set()
        has_active_filter = False
        for key in ["online_methods", "offline_methods"]:
            val = exp_cfg.get(key, [])
            if val:
                has_active_filter = True
                if isinstance(val, (list, tuple)):
                    methods = list(val)
                else:
                    methods = [item.strip() for item in str(val).split(",") if item.strip()]
                for m in methods:
                    active_aliases.update(get_method_aliases(m))

        method_ckpts = {}
        for method_dir in sorted(ckpt_root.iterdir()):
            if method_dir.is_dir():
                m_name = method_dir.name
                if has_active_filter and m_name not in active_aliases:
                    continue

                best_ckpt = None
                storage_url = exp_cfg.get("hydra", {}).get("sweeper", {}).get("storage", None)
                if storage_url:
                    from src.pipeline.optuna_utils import get_best_trial_id
                    study_name = f"{clean_exp}_{m_name}"
                    best_id = get_best_trial_id(storage_url, study_name)
                    candidate = method_dir / best_id / "best_model.ckpt"
                    if candidate.exists():
                        best_ckpt = candidate

                if not best_ckpt:
                    ckpts = list(method_dir.rglob("best_model*.ckpt"))
                    if ckpts:
                        best_ckpt = ckpts[0]

                if best_ckpt:
                    canon = get_canonical_method_name(m_name)
                    if canon not in method_ckpts or m_name == canon:
                        method_ckpts[canon] = best_ckpt
        return method_ckpts

    def _load_agent(self, path, dev):
        from src.methods.cql_agent import CQLAgent
        from src.methods.cew_agent import CEWAgent
        from src.methods.iql_agent import IQLAgent
        last_error = None
        for cls in [CQLAgent, CEWAgent, IQLAgent]:
            try:
                ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False)
                ag.to(dev)
                ag.eval()
                return ag
            except Exception as e:
                last_error = e
                try:
                    ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False, strict=False)
                    ag.to(dev)
                    ag.eval()
                    return ag
                except Exception as e2:
                    last_error = e2
                    continue
        if last_error is not None:
            print(f"  [clinical_alignment] Checkpoint load error for {path}: {last_error}")
        return None

    def _get_probs_and_actions(self, ag, obs_b):
        if hasattr(ag, "is_modular") and ag.is_modular:
            logic_obs = ag._prepare_logic_obs(obs_b) if hasattr(ag, "_prepare_logic_obs") else obs_b.unsqueeze(1).repeat(1, 2, 1)
            probs, _ = ag.model.actor(obs_b, logic_obs)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "actor") and hasattr(ag.actor, "get_action_probs"):
            probs = ag.actor.get_action_probs(obs_b)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "fuzzy_model") and ag.fuzzy_model is not None:
            q = ag.fuzzy_model(obs_b.to("cpu"))
            probs = torch.softmax(q, dim=-1).to(obs_b.device)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "q_network"):
            if hasattr(ag.q_network, "get_action_probs"):
                probs = ag.q_network.get_action_probs(obs_b)
            else:
                q = ag.q_network(obs_b)
                probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "model") and hasattr(ag.model, "get_q_values"):
            q = ag.model.get_q_values(obs_b)
            probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        else:
            out = ag.get_action_and_value(obs_b)
            act = out[0] if isinstance(out, (tuple, list)) else out
            n_acts = 3 if obs_b.shape[-1] >= 123 else 2
            probs = torch.zeros((obs_b.shape[0], n_acts), device=obs_b.device)
            probs.scatter_(1, act.unsqueeze(1).long(), 1.0)
            return probs, act

    def _run_mimic_eval(self, exp_id: str, cfg: dict, group: str, clean_exp: str, output_dir: Path):
        """Clinical policy alignment and septic shock evaluation for MIMIC datasets."""
        env_ds = cfg.get("env", {}).get("dataset_name", "mimic_lazy_0_interventions_balanced.npz") if isinstance(cfg.get("env"), dict) else "mimic_lazy_0_interventions_balanced.npz"
        npz_candidate = Path("in/datasets/mimic") / env_ds
        if not npz_candidate.exists():
            mode_path = cfg.get("mode", {}).get("dataset_path", "")
            if mode_path:
                cand = Path(mode_path).with_suffix(".npz")
                if cand.exists():
                    npz_candidate = cand

        if not npz_candidate.exists():
            raise FileNotFoundError(f"[clinical_alignment]: MIMIC dataset file '{npz_candidate}' not found.")

        print(f"\n==========================================================================================")
        print(f"=== Running MIMIC Clinical Alignment Evaluation for '{exp_id}' ===")
        print(f"==========================================================================================")
        data = np.load(npz_candidate, allow_pickle=True)
        X = data['X']        # (N, 240, 49)
        mask = data['mask']  # (N, 240, 1)

        valid_mask = (mask.squeeze(-1) != -1)
        all_obs = X[:, :, :46][valid_mask]
        all_clin_acts = X[:, :, 47][valid_mask].astype(int)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        total_steps = len(all_clin_acts)

        method_ckpts = self._discover_checkpoints(exp_id, group, clean_exp)
        if not method_ckpts:
            print(f"Notice [clinical_alignment]: No policy checkpoints found for '{clean_exp}'")
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
            "F1 Score": 1.0000,
            "Best F1": 1.0000,
            "Windowed F1 (±3h)": 1.0000,
            "Windowed Recall %": 100.0,
            "Opt Threshold": 0.5000
        })

        num_patients = X.shape[0]
        outcomes = data['y'].squeeze() if 'y' in data else np.zeros(num_patients)
        patient_agreements = {}

        batch_size = 10000
        for method_name, ckpt_path in sorted(method_ckpts.items()):
            agent = self._load_agent(ckpt_path, device)
            if agent is None:
                print(f"  Warning [clinical_alignment]: Could not load checkpoint {ckpt_path}")
                continue

            all_admin_probs = []
            all_policy_acts = []

            with torch.no_grad():
                for b_start in range(0, total_steps, batch_size):
                    b_end = min(b_start + batch_size, total_steps)
                    obs_batch = torch.tensor(all_obs[b_start:b_end], dtype=torch.float32).to(device)

                    probs, policy_acts_tensor = self._get_probs_and_actions(agent, obs_batch)
                    policy_acts = policy_acts_tensor.cpu().numpy()
                    if probs.shape[-1] > 1:
                        admin_probs = probs[:, 1].cpu().numpy()
                    else:
                        admin_probs = probs.squeeze().cpu().numpy()

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
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            try:
                auc_roc = float(roc_auc_score(all_clin_acts, all_admin_probs))
            except Exception:
                auc_roc = float('nan')

            try:
                auprc = float(average_precision_score(all_clin_acts, all_admin_probs))
            except Exception:
                auprc = float('nan')

            # Best F1 threshold sweep
            try:
                p_thresh, r_thresh, thresholds = precision_recall_curve(all_clin_acts, all_admin_probs)
                f1_scores = 2 * (p_thresh * r_thresh) / (p_thresh + r_thresh + 1e-8)
                best_idx = np.argmax(f1_scores)
                best_f1 = float(f1_scores[best_idx])
                opt_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
            except Exception:
                best_f1 = f1
                opt_thresh = 0.5

            # Windowed agreement calculation (±3 hours)
            step_idx = 0
            patient_agrs = []
            win_tp = 0
            win_fp = 0
            win_fn = 0
            total_clin_pos = 0

            for p_idx in range(num_patients):
                p_valid = valid_mask[p_idx]
                p_len = p_valid.sum()
                if p_len == 0:
                    continue

                p_clin = all_clin_acts[step_idx:step_idx + p_len]
                p_pol = all_policy_acts[step_idx:step_idx + p_len]
                step_idx += p_len

                p_agree = (p_clin == p_pol).mean()
                patient_agrs.append(p_agree)

                clin_pos_indices = np.where(p_clin == 1)[0]
                pol_pos_indices = np.where(p_pol == 1)[0]
                total_clin_pos += len(clin_pos_indices)

                for c_pos in clin_pos_indices:
                    if len(pol_pos_indices) > 0 and np.min(np.abs(pol_pos_indices - c_pos)) <= 3:
                        win_tp += 1
                    else:
                        win_fn += 1

                for p_pos in pol_pos_indices:
                    if len(clin_pos_indices) == 0 or np.min(np.abs(clin_pos_indices - p_pos)) > 3:
                        win_fp += 1

            patient_agreements[method_name] = np.array(patient_agrs)
            win_precision = win_tp / (win_tp + win_fp + 1e-8)
            win_recall = win_tp / (total_clin_pos + 1e-8)
            win_f1 = 2 * (win_precision * win_recall) / (win_precision + win_recall + 1e-8)

            results.append({
                "Method": clean_label(method_name),
                "Accuracy %": float(accuracy),
                "Admin Rate %": float(admin_rate),
                "AUC-ROC": auc_roc,
                "AUPRC": auprc,
                "Precision": float(precision),
                "Recall": float(recall),
                "F1 Score": float(f1),
                "Best F1": best_f1,
                "Windowed F1 (±3h)": float(win_f1),
                "Windowed Recall %": float(win_recall * 100.0),
                "Opt Threshold": opt_thresh
            })

        df = pd.DataFrame(results)
        csv_path = output_dir / "method_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved MIMIC method comparison: {csv_path}")

        # 1. Clinician Agreement % Bar Chart
        fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.8), 5.5))
        methods = [r["Method"] for r in results]
        accuracies = [r["Accuracy %"] for r in results]

        bar_colors = []
        for r in results:
            m_name = r["Method"]
            if "Clinician" in m_name:
                bar_colors.append("#7f7f7f")
            else:
                style = get_method_style(m_name)
                bar_colors.append(style.get("color") or "tab:blue")

        bars = ax.bar(methods, accuracies, color=bar_colors, width=0.55, edgecolor="#333333", linewidth=1.0, alpha=0.85)
        ax.set_ylabel("Clinician Agreement (%)", fontsize=12, fontweight="bold")
        ax.set_title(f"MIMIC Treatment Action Agreement ({clean_exp})", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=15, ha="right", fontsize=10, fontweight="bold")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig.tight_layout()
        plot_path = output_dir / "clinical_agreement.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"  Saved: {plot_path}")

        # 2. Clinician Agreement vs Septic Shock Outcome Analysis
        if patient_agreements and len(outcomes) == len(next(iter(patient_agreements.values()))):
            fig, ax = plt.subplots(figsize=(10, 6))
            for m_name, p_agr in patient_agreements.items():
                style = get_method_style(m_name)
                color = style.get("color") or "tab:blue"

                deciles = np.percentile(p_agr, np.linspace(0, 100, 11))
                decile_centers = []
                shock_rates = []

                for i in range(len(deciles) - 1):
                    low = deciles[i]
                    high = deciles[i+1]
                    m = (p_agr >= low) & (p_agr <= high)
                    if m.sum() > 0:
                        decile_centers.append((low + high) / 2.0 * 100.0)
                        shock_rates.append(outcomes[m].mean() * 100.0)

                if len(decile_centers) > 1:
                    ax.plot(decile_centers, shock_rates, marker="o", linewidth=2.2, label=clean_label(m_name), color=color)

            ax.set_xlabel("Clinician Agreement Rate (%)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Septic Shock Incidence (%)", fontsize=11, fontweight="bold")
            ax.set_title("Septic Shock Incidence vs Clinician Agreement Deciles", fontsize=13, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best", fontsize=10)
            fig.tight_layout()
            shock_plot_path = output_dir / "clinician_agreement_vs_shock.png"
            plt.savefig(shock_plot_path, dpi=200)
            plt.close()
            print(f"  Saved: {shock_plot_path}")

        # 3. Save Summary Markdown Report
        report_path = output_dir / "clinical_alignment_report.md"
        with open(report_path, "w") as f:
            f.write(f"# MIMIC Clinical Policy Alignment Report: `{clean_exp}`\n\n")
            f.write(pd.DataFrame(results).to_markdown(index=False))
            f.write("\n\n---\n*Auto-generated by NeSyRL Pipeline*\n")
        print(f"  Saved Markdown Report:     {report_path}")
        print("==========================================================================================\n")


