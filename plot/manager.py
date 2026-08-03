#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import yaml

from plot.base import BasePlotter
from plot.convergence import ConvergencePlotter
from plot.losses import LossesPlotter
from plot.reports import ReportsPlotter
from plot.early_prediction import EarlyPredictionPlotter

PLOTTER_REGISTRY = {
    "convergence": ConvergencePlotter,
    "losses": LossesPlotter,
    "reports": ReportsPlotter,
    "early_prediction": EarlyPredictionPlotter
}

def run_experiment_plots(exp_id: str):
    print(f"\n==================================================")
    print(f"=== Auto-Generating Plots for Experiment: {exp_id} ===")
    print(f"==================================================")

    exp_path = Path(f"in/config/experiment/{exp_id}.yaml")
    exp_cfg = {}
    if exp_path.exists():
        with open(exp_path) as f:
            exp_cfg = yaml.safe_load(f) or {}

    plots_req = exp_cfg.get("plots", ["convergence", "losses", "reports"])
    
    if isinstance(plots_req, list):
        requested_modules = {item: {} for item in plots_req}
    elif isinstance(plots_req, dict):
        requested_modules = plots_req
    else:
        requested_modules = {"convergence": {}, "losses": {}, "reports": {}}

    # Auto-include early_prediction plotter for MIMIC or early_prediction experiments
    env_name = exp_cfg.get("env", {}).get("name", "") if isinstance(exp_cfg.get("env"), dict) else ""
    task_name = exp_cfg.get("task", "")
    if "early_prediction" not in requested_modules:
        if env_name == "mimic" or "early_pred" in exp_id or "early_prediction" in task_name:
            requested_modules["early_prediction"] = {}

    for module_name, overrides in requested_modules.items():
        if module_name in PLOTTER_REGISTRY:
            plotter_cls = PLOTTER_REGISTRY[module_name]
            plotter = plotter_cls()
            try:
                plotter.run(exp_id, cli_overrides=overrides if isinstance(overrides, dict) else None)
            except Exception as e:
                print(f"Error running plotter '{module_name}' for '{exp_id}': {e}")
        else:
            print(f"Warning: Unknown plotter module '{module_name}' requested in experiment config.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate Experiment Plot Generation")
    parser.add_argument("experiment_id", type=str, help="Experiment ID to generate plots for")
    args = parser.parse_args()

    run_experiment_plots(args.experiment_id)
