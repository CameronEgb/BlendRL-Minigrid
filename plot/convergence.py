#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

from plot.base import BasePlotter, clean_label, get_style_info, moving_average

class ConvergencePlotter(BasePlotter):
    def __init__(self):
        super().__init__("convergence")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        metrics = cfg.get("metrics", ["eval/reward", "train/reward", "train/length"])
        window = cfg.get("smoothing_window", 10)
        dpi = cfg.get("dpi", 300)
        figsize = tuple(cfg.get("figsize", [8, 5]))
        x_axis_col = cfg.get("x_axis", "transitions")

        runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            print(f"No log data found for experiment '{exp_id}' in group '{group}'.")
            return

        conv_dir = output_dir / "convergence"
        conv_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Generating Convergence Plots for '{exp_id}' (Group: {group}) ===")

        for metric in metrics:
            plt.figure(figsize=figsize)
            has_data = False

            for method_name, versions in sorted(runs_data.items()):
                # Aggregate metrics across versions if multiple exist
                all_x = []
                all_y = []

                for v_name, df in versions.items():
                    if metric in df.columns:
                        valid_df = df.dropna(subset=[metric])
                        if not valid_df.empty:
                            x_vals = valid_df[x_axis_col].values if x_axis_col in valid_df.columns else valid_df.index.values
                            y_vals = valid_df[metric].values
                            all_x.append(x_vals)
                            all_y.append(y_vals)

                if all_y:
                    has_data = True
                    display_name = clean_label(method_name)
                    color, ls, marker = get_style_info(method_name)

                    # Simple mean if lengths match or plot single/first version
                    y_avg = moving_average(all_y[0], window)
                    x_avg = all_x[0][:len(y_avg)]

                    plt.plot(x_avg, y_avg, label=display_name, color=color, linestyle=ls, linewidth=2.0)

            if has_data:
                plt.xlabel(x_axis_col.replace("_", " ").title())
                plt.ylabel(metric.replace("_", " ").title())
                plt.title(f"{exp_id.upper()}: {metric}")
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()

                safe_metric_name = metric.replace("/", "_")
                out_path = conv_dir / f"{safe_metric_name}.png"
                plt.savefig(out_path, dpi=dpi)
                plt.close()
                print(f"  Saved: {out_path}")
            else:
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Convergence Plots for an Experiment")
    parser.add_argument("experiment_id", type=str, help="Experiment ID to plot")
    parser.add_argument("--metrics", nargs="+", help="Specific metrics to plot (e.g. eval/reward train/reward)")
    parser.add_argument("--window", type=int, help="Smoothing window size")
    parser.add_argument("--dpi", type=int, help="Plot resolution DPI")
    args = parser.parse_args()

    cli_overrides = {}
    if args.metrics:
        cli_overrides["metrics"] = args.metrics
    if args.window:
        cli_overrides["smoothing_window"] = args.window
    if args.dpi:
        cli_overrides["dpi"] = args.dpi

    plotter = ConvergencePlotter()
    plotter.run(args.experiment_id, cli_overrides)
