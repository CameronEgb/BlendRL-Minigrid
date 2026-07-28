#!/usr/bin/env python3
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

from plot.base import BasePlotter, clean_label

class ReportsPlotter(BasePlotter):
    def __init__(self):
        super().__init__("reports")

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
                f.write("| Method | Folder | Status |\n")
                f.write("| --- | --- | --- |\n")
                for method, versions in sorted(runs_data.items()):
                    f.write(f"| {clean_label(method)} | `{method}` | Completed |\n")
            print(f"  Saved: {hp_path}")

        if cfg.get("include_methods_comparison", True):
            comp_path = output_dir / "methods_comparison_report.md"
            with open(comp_path, "w") as f:
                f.write(f"# Methods Comparison Report: {exp_id}\n\n")
                f.write("| Method | Evaluation Metric | Final Value |\n")
                f.write("| --- | --- | --- |\n")
                for method, versions in sorted(runs_data.items()):
                    for v_name, df in versions.items():
                        if "eval/reward" in df.columns:
                            last_val = df["eval/reward"].dropna().iloc[-1] if not df["eval/reward"].dropna().empty else "N/A"
                            f.write(f"| {clean_label(method)} | eval/reward | {last_val:.2f} |\n")
            print(f"  Saved: {comp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown Reports for an Experiment")
    parser.add_argument("experiment_id", type=str, help="Experiment ID to generate reports for")
    args = parser.parse_args()

    plotter = ReportsPlotter()
    plotter.run(args.experiment_id)
