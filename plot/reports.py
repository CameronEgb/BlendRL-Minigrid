import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import pandas as pd
import json
import yaml
from typing import Optional

from plot.base import BasePlotter, clean_label


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "N/A"
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.2f}s"



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

        # Clean up legacy markdown comparison report if present
        old_report = output_dir / "methods_comparison_report.md"
        if old_report.exists():
            try:
                old_report.unlink()
            except Exception:
                pass

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

        if cfg.get("include_time_report", True):
            time_path = output_dir / "time_report.md"
            time_csv_path = output_dir / "time_report.csv"

            timing_rows = []

            for method, versions in sorted(runs_data.items()):
                method_label = clean_label(method)
                times = []

                for v_name, df in sorted(versions.items()):
                    t_sec = None
                    # 1. Try loading from runtime.json
                    v_log_dir = Path("results/logs") / group / exp_id / method / v_name
                    v_ckpt_dir = Path("results/checkpoints") / group / exp_id / method / v_name.replace("version_", "")

                    json_candidates = [
                        v_log_dir / "runtime.json",
                        v_ckpt_dir / "runtime.json",
                        Path("results/logs") / group / exp_id / method / "runtime.json",
                        Path("results/checkpoints") / group / exp_id / method / "runtime.json"
                    ]

                    for json_path in json_candidates:
                        if json_path.exists():
                            try:
                                with open(json_path) as jf:
                                    rdata = json.load(jf)
                                    t_sec = float(rdata.get("training_time_seconds", 0.0))
                                    if t_sec > 0:
                                        break
                            except Exception:
                                pass

                    # 2. Fall back to metrics.csv training_time_seconds if present
                    if (t_sec is None or t_sec == 0) and "training_time_seconds" in df.columns:
                        vals = df["training_time_seconds"].dropna()
                        if not vals.empty:
                            t_sec = float(vals.iloc[-1])

                    if t_sec is not None and t_sec > 0:
                        times.append((v_name, t_sec))

                if times:
                    avg_time = sum(t for _, t in times) / len(times)
                    formatted_avg = format_duration(avg_time)
                    timing_rows.append({
                        "method_raw": method,
                        "method": method_label,
                        "num_runs": len(times),
                        "avg_time_sec": avg_time,
                        "formatted_avg": formatted_avg,
                        "details": times
                    })
                else:
                    timing_rows.append({
                        "method_raw": method,
                        "method": method_label,
                        "num_runs": 0,
                        "avg_time_sec": None,
                        "formatted_avg": "N/A",
                        "details": []
                    })

            with open(time_path, "w") as f:
                f.write(f"# Execution Time Report: {exp_id}\n\n")
                f.write("| Method | Runs | Avg Training Time (s) | Formatted Time |\n")
                f.write("| --- | --- | --- | --- |\n")

                csv_records = []
                for row in timing_rows:
                    time_str = f"{row['avg_time_sec']:.2f}" if row['avg_time_sec'] is not None else "N/A"
                    f.write(f"| `{row['method']}` | {row['num_runs']} | {time_str} | {row['formatted_avg']} |\n")
                    csv_records.append({
                        "Method": row['method'],
                        "Raw_Method": row['method_raw'],
                        "Runs": row['num_runs'],
                        "Avg_Time_Seconds": row['avg_time_sec'] if row['avg_time_sec'] is not None else "",
                        "Formatted_Time": row['formatted_avg']
                    })
                f.write("\n")

            pd.DataFrame(csv_records).to_csv(time_csv_path, index=False)
            print(f"  Saved: {time_path}")
            print(f"  Saved: {time_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown Reports for an Experiment")
    parser.add_argument("experiment_id", type=str, help="Experiment ID to generate reports for")
    args = parser.parse_args()

    plotter = ReportsPlotter()
    plotter.run(args.experiment_id)
