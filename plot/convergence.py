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
        self.plot_metric_series(exp_id, group, output_dir, metrics, cfg)

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
