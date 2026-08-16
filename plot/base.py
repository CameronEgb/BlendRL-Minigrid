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

import glob
import re
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Import styling and alias resolution from the unified method registry
from src.method_registry import clean_label, get_style_info, get_canonical_method_name, get_method_aliases

def moving_average(a: np.ndarray, n: int = 5) -> np.ndarray:
    if len(a) == 0:
        return np.array([])
    n = min(len(a), max(1, n))
    a_padded = np.pad(a, (n - 1, 0), mode="edge")
    ret = np.cumsum(a_padded, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def deep_update(base: dict, update: dict) -> dict:
    """Recursively updates a nested dictionary."""
    result = dict(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
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

    def _resolve_config_defaults(self, raw_cfg: dict, visited: Optional[set] = None) -> dict:
        """Recursively resolves Hydra-style defaults entries."""
        if visited is None:
            visited = set()

        resolved = dict(raw_cfg)
        defaults = resolved.pop("defaults", None)
        if not defaults or not isinstance(defaults, list):
            return resolved

        base_acc = {}
        for item in defaults:
            if isinstance(item, str):
                item_str = item.strip()
                if item_str == "_self_":
                    continue
                # Handle override patterns
                if item_str.startswith("override /env:"):
                    env_name = item_str.split(":", 1)[1].strip()
                    env_path = Path(f"in/config/env/{env_name}.yaml")
                    if env_path.exists():
                        base_acc["env"] = self._load_yaml(env_path)
                    continue
                if item_str.startswith("override /agent:"):
                    agent_name = item_str.split(":", 1)[1].strip()
                    agent_path = Path(f"in/config/agent/{agent_name}.yaml")
                    if agent_path.exists():
                        base_acc["agent"] = self._load_yaml(agent_path)
                    continue
                # Experiment base config (e.g. mimic/_base or cartpole/_base)
                base_cands = [
                    Path(f"in/config/experiment/{item_str}.yaml"),
                    Path(f"in/config/experiment/{Path(item_str).stem}.yaml"),
                ]
                for bcand in base_cands:
                    if bcand.exists() and str(bcand) not in visited:
                        visited.add(str(bcand))
                        parent_raw = self._load_yaml(bcand)
                        parent_resolved = self._resolve_config_defaults(parent_raw, visited)
                        base_acc = deep_update(base_acc, parent_resolved)
                        break
            elif isinstance(item, dict):
                for k, v in item.items():
                    if k.startswith("override /env"):
                        env_path = Path(f"in/config/env/{v}.yaml")
                        if env_path.exists():
                            base_acc["env"] = self._load_yaml(env_path)
                    elif k.startswith("override /agent"):
                        agent_path = Path(f"in/config/agent/{v}.yaml")
                        if agent_path.exists():
                            base_acc["agent"] = self._load_yaml(agent_path)
                    elif isinstance(v, str):
                        base_cand = Path(f"in/config/experiment/{v}.yaml")
                        if base_cand.exists() and str(base_cand) not in visited:
                            visited.add(str(base_cand))
                            parent_raw = self._load_yaml(base_cand)
                            parent_resolved = self._resolve_config_defaults(parent_raw, visited)
                            base_acc = deep_update(base_acc, parent_resolved)

        return deep_update(base_acc, resolved)

    def get_experiment_config(self, exp_id: str, exp_config_name: Optional[str] = None) -> dict:
        """Finds, resolves, and loads the experiment configuration YAML or saved run config."""
        clean_exp = Path(exp_id).stem
        candidates = []
        if exp_config_name:
            clean_base = Path(exp_config_name).stem
            candidates.extend([
                Path(f"in/config/experiment/{exp_config_name}.yaml"),
                Path(f"in/config/experiment/{clean_base}.yaml"),
            ])
            candidates.extend(list(Path("in/config/experiment").glob(f"**/{clean_base}.yaml")))

        candidates.extend([
            Path(f"in/config/experiment/{exp_id}.yaml"),
            Path(f"in/config/experiment/{clean_exp}.yaml"),
        ])
        candidates.extend(list(Path("in/config/experiment").glob(f"**/{clean_exp}.yaml")))
        
        for cand in candidates:
            if cand.exists():
                raw = self._load_yaml(cand)
                return self._resolve_config_defaults(raw)

        # Check saved run configs in results/logs and results/checkpoints
        for base_dir in [Path("results/logs"), Path("results/checkpoints")]:
            if base_dir.exists():
                matches = list(base_dir.glob(f"*/{clean_exp}/config.yaml"))
                if matches:
                    return self._resolve_config_defaults(self._load_yaml(matches[0]))
                matches_nested = list(base_dir.glob(f"*/{clean_exp}/*/config.yaml"))
                if matches_nested:
                    return self._resolve_config_defaults(self._load_yaml(matches_nested[0]))

        return {}

    def get_group(self, exp_id: str, exp_config: dict) -> str:
        """Resolves group for the given experiment ID."""
        if "group" in exp_config and exp_config["group"]:
            return exp_config["group"]
        
        # If exp_id has a group prefix like mimic/mimic_test
        if "/" in exp_id:
            parts = exp_id.split("/")
            return parts[0]

        clean_exp = Path(exp_id).stem
        # Scan results/logs/*/clean_exp and results/checkpoints/*/clean_exp
        for base_dir in [Path("results/logs"), Path("results/checkpoints"), Path("results/plots")]:
            if base_dir.exists():
                for g_dir in base_dir.iterdir():
                    if g_dir.is_dir() and (g_dir / clean_exp).exists():
                        return g_dir.name
        return "ungrouped"

    def get_effective_config(self, exp_id: str, cli_overrides: Optional[dict] = None, exp_config_name: Optional[str] = None) -> Tuple[dict, str, Path]:
        """
        Merges default module config < default_cfg
               < experiment config plots.<module_name>
               < CLI overrides
        Returns (merged_config, group, output_dir).
        """
        exp_cfg = self.get_experiment_config(exp_id, exp_config_name=exp_config_name)
        group = self.get_group(exp_id, exp_cfg)
        clean_exp = Path(exp_id).stem

        # Extract per-plotter options from experiment YAML
        exp_plot_opts = {}
        plots_sec = exp_cfg.get("plots", {})
        if isinstance(plots_sec, dict) and self.name in plots_sec:
            if isinstance(plots_sec[self.name], dict):
                exp_plot_opts = plots_sec[self.name]

        merged = deep_update(self.default_cfg, exp_plot_opts)
        if cli_overrides:
            merged = deep_update(merged, cli_overrides)

        output_dir = Path("results/plots") / group / clean_exp
        output_dir.mkdir(parents=True, exist_ok=True)
        return merged, group, output_dir

    def load_metrics(self, group: str, exp_id: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Loads metrics.csv files for all runs matching results/logs/[group]/[exp_id]/[method]/*.
        Filters by active online_methods and offline_methods from experiment config if defined.
        Returns dict: { method_name: { version_str: df } }
        """
        clean_exp = Path(exp_id).stem
        exp_dir = Path("results/logs") / group / clean_exp
        if not exp_dir.exists():
            print(f"Warning: Log directory {exp_dir} not found.")
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

        results = {}
        for method_dir in sorted(exp_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            raw_method_name = method_dir.name
            if has_active_filter and raw_method_name not in active_aliases:
                continue

            canon_name = get_canonical_method_name(raw_method_name)
            if canon_name not in results:
                results[canon_name] = {}

            # Check version_X subdirectories
            version_dirs = sorted([d for d in method_dir.glob("version_*") if d.is_dir()])
            if not version_dirs:
                version_dirs = [method_dir]

            for v_dir in version_dirs:
                csv_path = v_dir / "metrics.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            v_key = f"{raw_method_name}_{v_dir.name}" if canon_name in results and v_dir.name in results[canon_name] else v_dir.name
                            results[canon_name][v_key] = df
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")

        # Filter out empty method entries
        return {k: v for k, v in results.items() if v}

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
        any_saved = False

        print(f"=== Generating {self.name.title()} Plots for '{exp_id}' ===")

        for metric in metrics:
            plt.figure(figsize=figsize)
            has_data = False
            used_xlabel = None
            for method_name, versions in sorted(runs_data.items()):
                all_x = []
                all_y = []

                for v_name, df in versions.items():
                    if metric in df.columns:
                        valid_df = df.dropna(subset=[metric])
                        if not valid_df.empty:
                            x_vals = None
                            if x_axis_col in df.columns and df[x_axis_col].notna().any():
                                full_x = df[x_axis_col].interpolate(method='linear').ffill().bfill()
                                s_x = full_x.loc[valid_df.index]
                                if not s_x.empty and s_x.nunique() > 1 and not s_x.isna().any():
                                    x_vals = s_x.values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", x_axis_col.replace("_", " ").title())
                            if x_vals is None:
                                if "step" in valid_df.columns and valid_df["step"].nunique() > 1:
                                    x_vals = valid_df["step"].values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", "Training Steps")
                                elif "epoch" in valid_df.columns and valid_df["epoch"].nunique() > 1:
                                    x_vals = valid_df["epoch"].values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", "Epoch")
                                else:
                                    x_vals = valid_df.index.values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", "Index")
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
                out_dir.mkdir(parents=True, exist_ok=True)
                plt.xlabel(used_xlabel or cfg.get("xlabel", "Training Steps"))
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
                any_saved = True
            else:
                plt.close()

        if out_dir.exists() and not any_saved and not any(out_dir.iterdir()):
            try:
                out_dir.rmdir()
            except Exception:
                pass

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        raise NotImplementedError("Subclasses must implement run()")
