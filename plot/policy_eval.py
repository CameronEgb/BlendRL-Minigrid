#!/usr/bin/env python3
import os
import sys

# Ensure project root and src are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
fyd_path = os.path.join(PROJECT_ROOT, "src", "fyd_repo", "src")
if fyd_path not in sys.path:
    sys.path.insert(0, fyd_path)

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from plot.base import BasePlotter, clean_label

class PolicyEvalPlotter(BasePlotter):
    def __init__(self):
        super().__init__("policy_eval")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        
        env_name = cfg.get("env", {}).get("name", "") if isinstance(cfg.get("env"), dict) else str(cfg.get("env", ""))
        
        if env_name == "pyrenees" or group == "pyrenees":
            self._run_pyrenees_eval(exp_id, cfg, group, clean_exp, output_dir)
        elif env_name == "mimic" or group == "mimic":
            self._run_mimic_eval(exp_id, cfg, group, clean_exp, output_dir)
        else:
            return

    def _discover_checkpoints(self, exp_id: str, group: str, clean_exp: str):
        ckpt_root = Path("results/checkpoints") / group / clean_exp
        if not ckpt_root.exists():
            ckpt_root = Path("results/checkpoints") / clean_exp
        if not ckpt_root.exists():
            return {}

        exp_cfg = self.get_experiment_config(exp_id)
        from plot.base import get_canonical_method_name, get_method_aliases
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
                ckpts = list(method_dir.rglob("best_model*.ckpt"))
                if ckpts:
                    canon = get_canonical_method_name(m_name)
                    if canon not in method_ckpts or m_name == canon:
                        method_ckpts[canon] = ckpts[0]
        return method_ckpts

    def _load_agent(self, path, dev):
        from src.methods.cql_agent import CQLAgent
        from src.methods.cew_agent import CEWAgent
        from src.methods.iql_agent import IQLAgent
        for cls in [CQLAgent, CEWAgent, IQLAgent]:
            try:
                ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False)
                ag.to(dev)
                ag.eval()
                return ag
            except Exception:
                continue
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

    def _run_pyrenees_eval(self, exp_id: str, cfg: dict, group: str, clean_exp: str, output_dir: Path):
        """Domain-appropriate policy evaluation for Pyrenees ITS stepwise transitions."""
        dataset_path = Path("in/datasets/pyrenees/pyrenees_clean.npz")
        if not dataset_path.exists():
            print(f"Notice [policy_eval]: Pyrenees dataset not found at {dataset_path}")
            return

        print(f"=== Running Pyrenees Policy Evaluation Module for '{exp_id}' ===")
        data = np.load(dataset_path, allow_pickle=True)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]

        all_obs = np.vstack(states)
        all_acts = np.hstack(actions).astype(int)
        all_rews = np.hstack(rewards).astype(float)
        total_steps = len(all_acts)

        # Competency segmentation if scaler available
        comp_tiers = np.ones(total_steps, dtype=int)  # Default Med
        scaler_path = Path("in/datasets/pyrenees/pyrenees_gmm_scaler.npz")
        if scaler_path.exists():
            try:
                gdata = np.load(scaler_path, allow_pickle=True)
                means = gdata["means"]
                precisions = gdata["precisions"]
                log_dets = gdata["log_dets"]
                log_weights = gdata["log_weights"]
                feat_idx = gdata["feature_indices"]

                x_feat = all_obs[:, feat_idx]
                d = x_feat.shape[-1]
                const = 0.5 * d * np.log(2.0 * np.pi)
                log_probs = []
                for k in range(len(means)):
                    diff = x_feat - means[k]
                    maha = np.sum(diff * (diff @ precisions[k]), axis=-1)
                    log_p = log_weights[k] - 0.5 * log_dets[k] - 0.5 * maha - const
                    log_probs.append(log_p)
                log_probs_mat = np.stack(log_probs, axis=-1)
                posteriors = np.exp(log_probs_mat - np.max(log_probs_mat, axis=-1, keepdims=True))
                posteriors /= posteriors.sum(axis=-1, keepdims=True)
                comp_tiers = posteriors.argmax(axis=1)
            except Exception as e:
                print(f"  Warning [policy_eval]: Could not compute GMM competency segmentation: {e}")

        method_ckpts = self._discover_checkpoints(exp_id, group, clean_exp)
        if not method_ckpts:
            print(f"Notice [policy_eval]: No policy checkpoints found for '{clean_exp}'")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        results = []

        # 1. Historical Tutor Baseline
        tutor_ps = (all_acts == 0).mean() * 100.0
        tutor_we = (all_acts == 1).mean() * 100.0
        tutor_fwe = (all_acts == 2).mean() * 100.0
        
        low_m = comp_tiers == 0
        med_m = comp_tiers == 1
        high_m = comp_tiers == 2

        results.append({
            "Method": "Historical Tutor (Baseline)",
            "Tutor Agreement %": 100.0,
            "Overall PS %": float(tutor_ps),
            "Overall WE %": float(tutor_we),
            "Overall FWE %": float(tutor_fwe),
            "Low Tier PS %": float((all_acts[low_m] == 0).mean() * 100.0) if low_m.sum() > 0 else 0.0,
            "Low Tier WE %": float((all_acts[low_m] == 1).mean() * 100.0) if low_m.sum() > 0 else 0.0,
            "Low Tier FWE %": float((all_acts[low_m] == 2).mean() * 100.0) if low_m.sum() > 0 else 0.0,
            "Med Tier PS %": float((all_acts[med_m] == 0).mean() * 100.0) if med_m.sum() > 0 else 0.0,
            "Med Tier WE %": float((all_acts[med_m] == 1).mean() * 100.0) if med_m.sum() > 0 else 0.0,
            "Med Tier FWE %": float((all_acts[med_m] == 2).mean() * 100.0) if med_m.sum() > 0 else 0.0,
            "High Tier PS %": float((all_acts[high_m] == 0).mean() * 100.0) if high_m.sum() > 0 else 0.0,
            "High Tier WE %": float((all_acts[high_m] == 1).mean() * 100.0) if high_m.sum() > 0 else 0.0,
            "High Tier FWE %": float((all_acts[high_m] == 2).mean() * 100.0) if high_m.sum() > 0 else 0.0,
            "Mean Step Reward": float(all_rews.mean()),
            "Agreed Step Reward": float(all_rews.mean()),
        })

        batch_size = 5000
        for method_name, ckpt_path in sorted(method_ckpts.items()):
            agent = self._load_agent(ckpt_path, device)
            if agent is None:
                print(f"  Warning [policy_eval]: Could not load checkpoint {ckpt_path}")
                continue

            all_policy_acts = []
            with torch.no_grad():
                for b_start in range(0, total_steps, batch_size):
                    b_end = min(b_start + batch_size, total_steps)
                    obs_batch = torch.tensor(all_obs[b_start:b_end], dtype=torch.float32).to(device)
                    _, policy_acts_tensor = self._get_probs_and_actions(agent, obs_batch)
                    all_policy_acts.extend(policy_acts_tensor.cpu().numpy())

            all_policy_acts = np.array(all_policy_acts)
            matches = (all_policy_acts == all_acts)
            agreement = matches.mean() * 100.0

            ps_rate = (all_policy_acts == 0).mean() * 100.0
            we_rate = (all_policy_acts == 1).mean() * 100.0
            fwe_rate = (all_policy_acts == 2).mean() * 100.0

            agreed_reward = float(all_rews[matches].mean()) if matches.sum() > 0 else float(all_rews.mean())

            results.append({
                "Method": clean_label(method_name),
                "Tutor Agreement %": float(agreement),
                "Overall PS %": float(ps_rate),
                "Overall WE %": float(we_rate),
                "Overall FWE %": float(fwe_rate),
                "Low Tier PS %": float((all_policy_acts[low_m] == 0).mean() * 100.0) if low_m.sum() > 0 else 0.0,
                "Low Tier WE %": float((all_policy_acts[low_m] == 1).mean() * 100.0) if low_m.sum() > 0 else 0.0,
                "Low Tier FWE %": float((all_policy_acts[low_m] == 2).mean() * 100.0) if low_m.sum() > 0 else 0.0,
                "Med Tier PS %": float((all_policy_acts[med_m] == 0).mean() * 100.0) if med_m.sum() > 0 else 0.0,
                "Med Tier WE %": float((all_policy_acts[med_m] == 1).mean() * 100.0) if med_m.sum() > 0 else 0.0,
                "Med Tier FWE %": float((all_policy_acts[med_m] == 2).mean() * 100.0) if med_m.sum() > 0 else 0.0,
                "High Tier PS %": float((all_policy_acts[high_m] == 0).mean() * 100.0) if high_m.sum() > 0 else 0.0,
                "High Tier WE %": float((all_policy_acts[high_m] == 1).mean() * 100.0) if high_m.sum() > 0 else 0.0,
                "High Tier FWE %": float((all_policy_acts[high_m] == 2).mean() * 100.0) if high_m.sum() > 0 else 0.0,
                "Mean Step Reward": float(all_rews.mean()),
                "Agreed Step Reward": agreed_reward,
            })

        df = pd.DataFrame(results)
        csv_path = output_dir / "method_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved Pyrenees method comparison: {csv_path}")

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
            print(f"Notice [policy_eval]: MIMIC NPZ dataset '{npz_candidate}' not found, skipping policy eval metrics.")
            return

        print(f"=== Running MIMIC Policy Evaluation Module for '{exp_id}' ===")
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
            print(f"Notice [policy_eval]: No policy checkpoints found for '{clean_exp}'")
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

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.method_registry import get_style as get_method_style

        num_patients = X.shape[0]
        outcomes = data['y'].squeeze() if 'y' in data else np.zeros(num_patients)
        patient_agreements = {}

        batch_size = 10000
        for method_name, ckpt_path in sorted(method_ckpts.items()):
            agent = self._load_agent(ckpt_path, device)
            if agent is None:
                print(f"  Warning [policy_eval]: Could not load checkpoint {ckpt_path}")
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
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            auc_roc = roc_auc_score(all_clin_acts, all_admin_probs) if len(np.unique(all_clin_acts)) > 1 else 0.0
            auprc = average_precision_score(all_clin_acts, all_admin_probs) if len(np.unique(all_clin_acts)) > 1 else 0.0

            if len(np.unique(all_clin_acts)) > 1 and len(all_admin_probs) > 0:
                p_curve, r_curve, th_curve = precision_recall_curve(all_clin_acts, all_admin_probs)
                f1_curve = 2 * (p_curve * r_curve) / (p_curve + r_curve + 1e-8)
                best_idx = np.argmax(f1_curve)
                best_f1 = float(f1_curve[best_idx])
                best_thresh = float(th_curve[best_idx]) if best_idx < len(th_curve) else 0.5
            else:
                best_f1, best_thresh = float(f1), 0.5

            patient_agr = []
            curr_step = 0
            win_tp = 0
            win_fp = 0
            win_fn = 0
            window_size = 3

            for i in range(num_patients):
                v_steps = int((mask[i, :, 0] != -1).sum())
                if v_steps == 0:
                    patient_agr.append(np.nan)
                    continue
                p_policy = all_policy_acts[curr_step:curr_step + v_steps]
                p_clin = all_clin_acts[curr_step:curr_step + v_steps]
                curr_step += v_steps
                agr = (p_policy == p_clin).mean() * 100.0
                patient_agr.append(agr)

                for t in range(v_steps):
                    if p_policy[t] == 1:
                        w_start = max(0, t - window_size)
                        w_end = min(v_steps, t + window_size + 1)
                        if (p_clin[w_start:w_end] == 1).any():
                            win_tp += 1
                        else:
                            win_fp += 1
                
                for t in range(v_steps):
                    if p_clin[t] == 1:
                        w_start = max(0, t - window_size)
                        w_end = min(v_steps, t + window_size + 1)
                        if not (p_policy[w_start:w_end] == 1).any():
                            win_fn += 1

            patient_agreements[method_name] = np.array(patient_agr)

            total_pos_clin = all_clin_acts.sum()
            win_prec = win_tp / (win_tp + win_fp + 1e-8)
            win_rec = (total_pos_clin - win_fn) / (total_pos_clin + 1e-8) if total_pos_clin > 0 else 0.0
            win_rec = max(0.0, float(win_rec))
            windowed_f1 = float(2 * win_prec * win_rec / (win_prec + win_rec + 1e-8))

            results.append({
                "Method": clean_label(method_name),
                "Accuracy %": float(accuracy),
                "Admin Rate %": float(admin_rate),
                "AUC-ROC": float(auc_roc),
                "AUPRC": float(auprc),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1 Score": float(f1),
                "Best F1": float(best_f1),
                "Windowed F1 (±3h)": float(windowed_f1),
                "Windowed Recall %": float(win_rec * 100.0),
                "Opt Threshold": float(best_thresh)
            })

        df = pd.DataFrame(results)
        csv_path = output_dir / "method_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved MIMIC method comparison: {csv_path}")

        # Generate Agreement vs Shock Rate Plot only for MIMIC
        if patient_agreements and len(outcomes) > 0:
            bins = np.linspace(0, 100, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2.0

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()

            method_names = list(patient_agreements.items())
            K = len(method_names)
            total_bar_width = 7.5
            bar_width = total_bar_width / max(K, 1)

            for k_idx, (method_name, agr) in enumerate(method_names):
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

                means_arr = np.array(means)
                sems_arr = np.array(sems)
                valid = ~np.isnan(means_arr)

                style = get_method_style(method_name)
                label = style["label"]
                color = style["color"]
                marker = style["marker"]
                linestyle = style.get("linestyle", "-")

                offset = (k_idx - (K - 1) / 2.0) * bar_width
                bar_x = bin_centers + offset
                ax2.bar(bar_x, counts, width=bar_width * 0.9, color=color, alpha=0.18,
                        edgecolor=color, linewidth=0.8, zorder=1)

                ax1.plot(bin_centers[valid], means_arr[valid], marker=marker, color=color,
                         linestyle=linestyle, label=label, linewidth=2.5, markersize=7, zorder=3)
                ax1.fill_between(bin_centers[valid],
                                 means_arr[valid] - sems_arr[valid],
                                 means_arr[valid] + sems_arr[valid],
                                 color=color, alpha=0.12, zorder=2)

            ax1.set_xlabel("Clinician – RL Policy Agreement (%)", fontsize=12, fontweight="bold")
            ax1.set_ylabel("True Septic Shock Rate (%)", fontsize=12, fontweight="bold")
            ax1.set_xticks(np.arange(0, 101, 10))
            ax1.set_xlim(-2, 102)
            ax1.grid(True, linestyle="--", alpha=0.4, zorder=0)

            ax2.set_ylabel("Patient Trajectory Count (Histogram)", fontsize=12, fontweight="bold", color="#555555")
            ax2.tick_params(axis='y', labelcolor="#555555")

            lines1, labels1 = ax1.get_legend_handles_labels()
            ax1.legend(lines1, labels1, fontsize=10, loc="best", framealpha=0.9)
            ax1.set_title(f"Septic Shock Rate vs. Clinician Agreement ({clean_exp})", fontsize=13, fontweight="bold")

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
