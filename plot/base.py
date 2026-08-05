import os
import sys
import glob
import re
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Import styling from the unified method registry
from src.method_registry import clean_label, get_style_info

def moving_average(a: np.ndarray, n: int = 5) -> np.ndarray:
    if len(a) == 0: return np.array([])
    n = min(len(a), max(1, n))
    a_padded = np.pad(a, (n-1, 0), mode='edge')
    ret = np.cumsum(a_padded, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def deep_update(source: dict, overrides: dict) -> dict:
    """Recursively updates dictionary source with overrides."""
    result = dict(source)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result

class BasePlotter:
    name: str = "base"

    def __init__(self, name: str):
        self.name = name
        self.plot_dir = Path(__file__).parent
        self.default_config_path = self.plot_dir / f"_{name}_config.yaml"
        self.default_cfg = self._load_yaml(self.default_config_path)

    def _load_yaml(self, path: Path) -> dict:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_experiment_config(self, exp_id: str) -> dict:
        """Finds and loads the experiment configuration YAML."""
        exp_path = Path(f"in/config/experiment/{exp_id}.yaml")
        if exp_path.exists():
            return self._load_yaml(exp_path)
        # Search recursively
        matches = list(Path("in/config/experiment").glob(f"**/{exp_id}.yaml"))
        if matches:
            return self._load_yaml(matches[0])
        return {}

    def get_group(self, exp_id: str, exp_config: dict) -> str:
        """Resolves group for the given experiment ID."""
        if "group" in exp_config and exp_config["group"]:
            return exp_config["group"]
        
        # Scan results/logs/*/exp_id
        logs_base = Path("results/logs")
        if logs_base.exists():
            for g_dir in logs_base.iterdir():
                if g_dir.is_dir() and (g_dir / exp_id).exists():
                    return g_dir.name
        return "ungrouped"

    def get_effective_config(self, exp_id: str, cli_overrides: Optional[dict] = None) -> Tuple[dict, str, Path]:
        """
        Merges default module config < default_cfg
               < experiment config plots.<module_name>
               < CLI overrides
        Returns (merged_config, group, output_dir).
        """
        exp_cfg = self.get_experiment_config(exp_id)
        group = self.get_group(exp_id, exp_cfg)

        # Extract per-plotter options from experiment YAML
        exp_plot_opts = {}
        plots_sec = exp_cfg.get("plots", {})
        if isinstance(plots_sec, dict) and self.name in plots_sec:
            if isinstance(plots_sec[self.name], dict):
                exp_plot_opts = plots_sec[self.name]

        merged = deep_update(self.default_cfg, exp_plot_opts)
        if cli_overrides:
            merged = deep_update(merged, cli_overrides)

        output_dir = Path("results/plots") / group / exp_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return merged, group, output_dir

    def load_metrics(self, group: str, exp_id: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Loads metrics.csv files for all runs matching results/logs/[group]/[exp_id]/[method]/*.
        Returns dict: { method_name: { version_str: df } }
        """
        exp_dir = Path("results/logs") / group / exp_id
        if not exp_dir.exists():
            print(f"Warning: Log directory {exp_dir} not found.")
            return {}

        results = {}
        for method_dir in exp_dir.iterdir():
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name
            results[method_name] = {}
            
            # Check version_X subdirectories
            version_dirs = [d for d in method_dir.glob("version_*") if d.is_dir()]
            if not version_dirs:
                version_dirs = [method_dir]
                
            for v_dir in version_dirs:
                csv_path = v_dir / "metrics.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        results[method_name][v_dir.name] = df
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
        return results

    def plot_metric_series(self, exp_id: str, group: str, output_dir: Path, 
                           metrics: list, cfg: dict):
        """Standard multi-method metric plotting with multi-version mean±SEM.
        
        Shared implementation used by ConvergencePlotter, LossesPlotter, and
        any future plotter that plots time-series metrics from metrics.csv.
        """
        runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            print(f"No log data found for experiment '{exp_id}' in group '{group}'.")
            return

        window = cfg.get("smoothing_window", 10)
        dpi = cfg.get("dpi", 300)
        figsize = tuple(cfg.get("figsize", [8, 5]))
        x_axis_col = cfg.get("x_axis", "transitions")
        subdir = cfg.get("output_subdir", self.name)
        filename_prefix = cfg.get("filename_prefix", "")

        out_dir = output_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Generating {self.name.title()} Plots for '{exp_id}' ===")

        for metric in metrics:
            plt.figure(figsize=figsize)
            has_data = False

            for method_name, versions in sorted(runs_data.items()):
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

                    if len(all_y) > 1:
                        # Multi-version: compute mean ± SEM across versions
                        min_len = min(len(y) for y in all_y)
                        trimmed = np.array([y[:min_len] for y in all_y])
                        y_mean = np.mean(trimmed, axis=0)
                        y_sem = np.std(trimmed, axis=0) / np.sqrt(len(all_y))
                        y_smoothed = moving_average(y_mean, window)
                        sem_smoothed = moving_average(y_sem, window)
                        x_plot = all_x[0][:len(y_smoothed)]
                        plt.plot(x_plot, y_smoothed, label=display_name, color=color,
                                 linestyle=ls, linewidth=2.0)
                        plt.fill_between(x_plot, y_smoothed - sem_smoothed,
                                         y_smoothed + sem_smoothed, color=color, alpha=0.15)
                    else:
                        # Single version: simple moving average
                        y_smoothed = moving_average(all_y[0], window)
                        x_plot = all_x[0][:len(y_smoothed)]
                        plt.plot(x_plot, y_smoothed, label=display_name, color=color,
                                 linestyle=ls, linewidth=2.0)

            if has_data:
                plt.xlabel(cfg.get("xlabel", x_axis_col.replace("_", " ").title()))
                plt.ylabel(cfg.get("ylabel", metric.replace("_", " ").title()))
                plt.title(f"{exp_id.upper()}: {metric}")
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()

                safe_metric_name = metric.replace("/", "_")
                out_path = out_dir / f"{filename_prefix}{safe_metric_name}.png"
                plt.savefig(out_path, dpi=dpi)
                plt.close()
                print(f"  Saved: {out_path}")
            else:
                plt.close()

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        raise NotImplementedError("Subclasses must implement run()")
