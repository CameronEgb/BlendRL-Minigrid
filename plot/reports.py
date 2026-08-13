#!/usr/bin/env python3
import sys
import argparse
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional

from plot.base import BasePlotter, clean_label


class ReportsPlotter(BasePlotter):
    def __init__(self):
        super().__init__("reports")

    def _find_hydra_config(self, group: str, exp_id: str, method: str) -> dict:
        """Attempt to load the resolved Hydra config for a specific method run.
        Searches the most recent hydra output directory that matches this experiment."""
        hydra_base = Path("results/hydra/outputs")
        if not hydra_base.exists():
            return {}

        # Walk date dirs in reverse chronological order
        for date_dir in sorted(hydra_base.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            for time_dir in sorted(date_dir.iterdir(), reverse=True):
                cfg_path = time_dir / ".hydra" / "config.yaml"
                if cfg_path.exists():
                    try:
                        with open(cfg_path) as f:
                            run_cfg = yaml.safe_load(f) or {}
                        # Check if this run matches our experiment and method
                        if (run_cfg.get("experiment_id") == exp_id and
                                run_cfg.get("agent", {}).get("name", "") == method):
                            return run_cfg
                    except Exception:
                        continue
        return {}

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            return

        print(f"=== Generating Markdown Reports for '{exp_id}' ===")

        if cfg.get("include_hyperparameters", True):
            hp_path = output_dir / "hyperparameters_report.md"
            with open(hp_path, "w") as f:
                f.write(f"# Hyperparameter Report: {exp_id}\n\n")
                for method, versions in sorted(runs_data.items()):
                    f.write(f"## {clean_label(method)} (`{method}`)\n\n")

                    # Try to load actual hyperparameters from Hydra config
                    hydra_cfg = self._find_hydra_config(group, exp_id, method)
                    agent_cfg = hydra_cfg.get("agent", {})

                    if agent_cfg:
                        # Extract key hyperparameters
                        hp_keys = ["lr", "gamma", "tau", "beta", "cql_alpha",
                                   "batch_size", "hidden_sizes", "epochs_per_interval",
                                   "blend_ent_coef", "logic_lr", "actor_mode", "blender_mode"]
                        f.write("| Parameter | Value |\n")
                        f.write("| --- | --- |\n")
                        for key in hp_keys:
                            if key in agent_cfg and agent_cfg[key] is not None:
                                f.write(f"| `{key}` | `{agent_cfg[key]}` |\n")
                        # Also include any remaining agent config keys not in the standard list
                        extra_keys = set(agent_cfg.keys()) - set(hp_keys) - {"name", "modules", "_target_"}
                        for key in sorted(extra_keys):
                            val = agent_cfg[key]
                            if val is not None and not isinstance(val, (dict, list)):
                                f.write(f"| `{key}` | `{val}` |\n")
                        f.write("\n")
                    else:
                        f.write("_No Hydra config found for this run._\n\n")

            print(f"  Saved: {hp_path}")

        if cfg.get("include_methods_comparison", True):
            comp_path = output_dir / "methods_comparison_report.md"
            with open(comp_path, "w") as f:
                f.write(f"# Methods Comparison Report: {exp_id}\n\n")

                # Check if counterfactual summary CSV exists
                cf_csv = output_dir / "counterfactual_summary.csv"
                if not cf_csv.exists():
                    cf_csv = Path("results/checkpoints") / group / exp_id / "counterfactual_summary.csv"
                
                if cf_csv.exists():
                    try:
                        cf_df = pd.read_csv(cf_csv)
                        f.write("## Counterfactual & Policy Alignment Evaluation\n\n")
                        f.write("| Method | Accuracy / Clinician Agr % | Admin Rate % | Precision | Recall | F1 Score | Pred Mortality % |\n")
                        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
                        for _, row in cf_df.iterrows():
                            m_name = clean_label(str(row["method"]))
                            agr = f"{float(row.get('agreement_mean', 0))*100:.2f}%" if "agreement_mean" in row else "N/A"
                            admin = f"{float(row.get('admin_rate_mean', 0))*100:.2f}%" if "admin_rate_mean" in row else "N/A"
                            prec = f"{float(row.get('precision_mean', 0)):.4f}" if "precision_mean" in row else "N/A"
                            rec = f"{float(row.get('recall_mean', 0)):.4f}" if "recall_mean" in row else "N/A"
                            f1 = f"{float(row.get('f1_mean', 0)):.4f}" if "f1_mean" in row else "N/A"
                            mort = f"{float(row.get('pred_mortality_mean', 0))*100:.2f}%" if "pred_mortality_mean" in row else "N/A"
                            f.write(f"| `{m_name}` | {agr} | {admin} | {prec} | {rec} | {f1} | {mort} |\n")
                        f.write("\n")
                    except Exception as e:
                        print(f"Notice: Could not embed counterfactual CSV in report: {e}")

                # Determine which evaluation metrics exist across all methods
                eval_metrics = set()
                for method, versions in runs_data.items():
                    for v_name, df in versions.items():
                        eval_metrics.update(c for c in df.columns if c.startswith("eval/"))

                eval_metrics = sorted(eval_metrics)
                if eval_metrics:
                    f.write("## Training & Checkpoint Metrics\n\n")
                    f.write("| Method | " + " | ".join(eval_metrics) + " |\n")
                    f.write("| --- | " + " | ".join(["---"] * len(eval_metrics)) + " |\n")
                    for method, versions in sorted(runs_data.items()):
                        row = [clean_label(method)]
                        latest_df = list(versions.values())[-1] if versions else pd.DataFrame()
                        for metric in eval_metrics:
                            if metric in latest_df.columns:
                                vals = latest_df[metric].dropna()
                                if not vals.empty:
                                    row.append(f"{vals.iloc[-1]:.4f}")
                                else:
                                    row.append("N/A")
                            else:
                                row.append("N/A")
                        f.write("| " + " | ".join(row) + " |\n")
            print(f"  Saved: {comp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown Reports for an Experiment")
    parser.add_argument("experiment_id", type=str, help="Experiment ID to generate reports for")
    args = parser.parse_args()

    plotter = ReportsPlotter()
    plotter.run(args.experiment_id)
