#!/usr/bin/env python3
import sys
import os
import json
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any

from plot.base import BasePlotter, clean_label

DISP_MAP = {
    "lstm_no_v": "LSTM (no V)",
    "lstm_with_v": "LSTM (with V)",
    "transformer_no_v": "Transformer (no V)",
    "transformer_with_v": "Transformer (with V)"
}

class EarlyPredictionPlotter(BasePlotter):
    def __init__(self):
        super().__init__("early_prediction")

    def plot_dl_sweep_results(self, output_dir: Path) -> bool:
        """Finds metrics_*.json files in output_dir (or subdirectories) and generates
        consolidated 4-panel and 2-panel DL sweep graphs + text summaries.
        """
        json_files = list(output_dir.glob("metrics_*.json"))
        if not json_files:
            # Search one level up or under early_prediction
            json_files = list(output_dir.rglob("metrics_*.json"))

        if not json_files:
            return False

        all_results: Dict[str, Any] = {}
        for json_file in json_files:
            if json_file.stat().st_size == 0:
                continue
            try:
                m_key = json_file.stem.replace("metrics_", "")
                disp_name = DISP_MAP.get(m_key, m_key)
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if data.get("tau"):
                        all_results[disp_name] = data
            except Exception as e:
                print(f"Warning loading {json_file}: {e}")

        if not all_results:
            return False

        print(f"\n--- Generating DL Sweep Consolidated Plots ({len(all_results)} model configs) ---")

        # 1. Save text summary table
        results_txt_path = output_dir / "results.txt"
        with open(results_txt_path, "w") as f:
            f.write(f"=== Septic Shock Early Prediction DL Sweep Results ===\n\n")
            for m_name, res_data in sorted(all_results.items()):
                f.write(f"Model Configuration: {m_name}\n")
                f.write(f"  Taus:   {res_data.get('tau')}\n")
                f.write(f"  AUCs:   {res_data.get('auc')} (SEMs: {res_data.get('auc_sem')})\n")
                f.write(f"  AUPRCs: {res_data.get('auprc')} (SEMs: {res_data.get('auprc_sem')})\n")
                f.write(f"  F1_opt: {res_data.get('f1_opt')} (SEMs: {res_data.get('f1_opt_sem')})\n\n")
        print(f"  Saved summary: {results_txt_path}")

        # Colors and markers for plots
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        markers = ['o', 's', '^', 'D', 'v', 'P']
        model_keys_sorted = sorted(all_results.keys())

        # 2. Generate 4-panel comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: AUC-ROC
        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res["auc"])
            sem_arr = np.array(res["auc_sem"])
            c_idx = idx % len(colors)
            axes[0, 0].plot(tau_arr, mean_arr, marker=markers[c_idx], color=colors[c_idx], label=m_name, linewidth=2)
            axes[0, 0].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[c_idx], alpha=0.15)
        axes[0, 0].set_title("AUC-ROC vs. Lead Time (τ)", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("Lead Time (hours early - τ)", fontsize=11)
        axes[0, 0].set_ylabel("AUC-ROC", fontsize=11)
        axes[0, 0].grid(True, linestyle="--", alpha=0.6)
        axes[0, 0].legend(fontsize=10)

        # Plot 2: AUPRC
        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res["auprc"])
            sem_arr = np.array(res["auprc_sem"])
            c_idx = idx % len(colors)
            axes[0, 1].plot(tau_arr, mean_arr, marker=markers[c_idx], color=colors[c_idx], label=m_name, linewidth=2)
            axes[0, 1].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[c_idx], alpha=0.15)
        axes[0, 1].set_title("AUPRC (PR-AUC) vs. Lead Time (τ)", fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel("Lead Time (hours early - τ)", fontsize=11)
        axes[0, 1].set_ylabel("AUPRC", fontsize=11)
        axes[0, 1].grid(True, linestyle="--", alpha=0.6)
        axes[0, 1].legend(fontsize=10)

        # Plot 3: F1-Opt
        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res["f1_opt"])
            sem_arr = np.array(res["f1_opt_sem"])
            c_idx = idx % len(colors)
            axes[1, 0].plot(tau_arr, mean_arr, marker=markers[c_idx], color=colors[c_idx], label=m_name, linewidth=2)
            axes[1, 0].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[c_idx], alpha=0.15)
        axes[1, 0].set_title("Optimal F1-Score (θ*) vs. Lead Time (τ)", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("Lead Time (hours early - τ)", fontsize=11)
        axes[1, 0].set_ylabel("Optimal F1-Score", fontsize=11)
        axes[1, 0].grid(True, linestyle="--", alpha=0.6)
        axes[1, 0].legend(fontsize=10)

        # Plot 4: F1 at 0.5 Threshold
        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res.get("f1_05", res.get("f1_opt")))
            sem_arr = np.array(res.get("f1_05_sem", res.get("f1_opt_sem")))
            c_idx = idx % len(colors)
            axes[1, 1].plot(tau_arr, mean_arr, marker=markers[c_idx], color=colors[c_idx], label=m_name, linewidth=2)
            axes[1, 1].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[c_idx], alpha=0.15)
        axes[1, 1].set_title("Standard F1-Score (θ=0.5) vs. Lead Time (τ)", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Lead Time (hours early - τ)", fontsize=11)
        axes[1, 1].set_ylabel("F1-Score (θ=0.5)", fontsize=11)
        axes[1, 1].grid(True, linestyle="--", alpha=0.6)
        axes[1, 1].legend(fontsize=10)

        plt.tight_layout()
        plot_4panel_path = output_dir / "4panel.png"
        plt.savefig(plot_4panel_path, dpi=200)
        plt.close()
        print(f"  Saved 4-panel sweep plot: {plot_4panel_path}")

        # 3. Generate 2-panel comparison plot
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res["auc"])
            sem_arr = np.array(res["auc_sem"])
            c_idx = idx % len(colors)
            axes2[0].plot(tau_arr, mean_arr, marker=markers[c_idx], color=colors[c_idx], label=m_name, linewidth=2)
            axes2[0].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[c_idx], alpha=0.15)
        axes2[0].set_title("AUC-ROC vs. Lead Time (τ)", fontsize=13, fontweight='bold')
        axes2[0].set_xlabel("Lead Time (hours early - τ)", fontsize=12)
        axes2[0].set_ylabel("AUC-ROC", fontsize=12)
        axes2[0].grid(True, linestyle="--", alpha=0.6)
        axes2[0].legend(fontsize=10)

        for idx, m_name in enumerate(model_keys_sorted):
            res = all_results[m_name]
            tau_arr = np.array(res["tau"])
            mean_arr = np.array(res["f1_opt"])
            sem_arr = np.array(res["f1_opt_sem"])
            c_idx = idx % len(colors)
            axes2[1].plot(tau_arr, mean_arr, marker=markers[c_idx], color=colors[c_idx], label=m_name, linewidth=2)
            axes2[1].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[c_idx], alpha=0.15)
        axes2[1].set_title("Optimal F1-Score vs. Lead Time (θ*)", fontsize=13, fontweight='bold')
        axes2[1].set_xlabel("Lead Time (hours early - τ)", fontsize=12)
        axes2[1].set_ylabel("F1-Score (θ*)", fontsize=12)
        axes2[1].grid(True, linestyle="--", alpha=0.6)
        axes2[1].legend(fontsize=10)

        plt.tight_layout()
        plot_2panel_path = output_dir / "2panel.png"
        plt.savefig(plot_2panel_path, dpi=200)
        plt.close()
        print(f"  Saved 2-panel sweep plot: {plot_2panel_path}")

        return True

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        
        if not cfg.get("enabled", True):
            print(f"EarlyPredictionPlotter disabled in config for '{exp_id}'. Skipping.")
            return

        print(f"\n==================================================")
        print(f"=== Generating Early Prediction Plots for '{exp_id}' ===")
        print(f"==================================================")

        # Step 1: Check and generate DL Model Sweep Plots if sweep metric files exist
        has_sweep_plots = self.plot_dl_sweep_results(output_dir)

        # Step 2: Check for RL policy checkpoints to generate evaluation plots (Agreement vs Shock, Tau vs Pred Shock)
        ckpt_dir = Path("results/checkpoints") / group / exp_id
        if not ckpt_dir.exists():
            matches = list(Path("results/checkpoints").glob(f"**/{exp_id}"))
            if matches:
                ckpt_dir = matches[0]

        if ckpt_dir.exists() and any(ckpt_dir.glob("**/*.ckpt")):
            print(f"  Found policy checkpoints under: {ckpt_dir}")
            project_root = Path(__file__).parent.parent
            venv_python = project_root / "venv" / "bin" / "python3"
            python_exe = str(venv_python) if venv_python.exists() else sys.executable

            cmd = [
                python_exe, "src/early_prediction/eval.py",
                "--checkpoint", str(ckpt_dir),
                "--output-dir", str(output_dir),
            ]

            if cfg.get("dataset_path"):
                cmd.extend(["--dataset-path", str(cfg["dataset_path"])])
            if cfg.get("ep_ckpt_root"):
                cmd.extend(["--ep-ckpt-root", str(cfg["ep_ckpt_root"])])
            if cfg.get("n_splits"):
                cmd.extend(["--n-splits", str(cfg["n_splits"])])
            if cfg.get("remake", False):
                cmd.append("--remake")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root) + ":" + str(project_root / "src") + ":" + env.get("PYTHONPATH", "")

            try:
                print(f"  Running Policy Early Prediction Evaluation: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, env=env)
                print(f"  Successfully generated Policy EP Evaluation plots in: {output_dir}")
            except subprocess.CalledProcessError as e:
                print(f"Error running Policy EP Evaluation for '{exp_id}': {e}")
        elif not has_sweep_plots:
            print(f"Warning: EarlyPredictionPlotter found no sweep metrics or policy checkpoints for experiment '{exp_id}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Early Prediction Plots for an Experiment")
    parser.add_argument("experiment_id", type=str, help="Experiment ID to generate plots for")
    args = parser.parse_args()

    plotter = EarlyPredictionPlotter()
    plotter.run(args.experiment_id)
