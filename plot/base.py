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

def clean_label(label: str) -> str:
    """Cleans up technical folder names into readable legend labels."""
    l = label
    l_lower = l.lower()

    if "fyd" in l_lower:
        return "CEW+FYD"
    if ("cew_only" in l_lower or l_lower in ["cew", "cew_only"] or "blendrl_cql_cew_only" in l_lower or l_lower == "blendrl_cql_cew") and "human" not in l_lower:
        return "CEW"

    l = l.replace("multi_arch_", "")
    
    if "blendrl_cql" in l.lower():
        suffix = l.lower().split("blendrl_cql_")[-1]
        if "human_cew" in suffix:
            suffix_str = "Human+CEW"
        elif "human_neural" in suffix:
            suffix_str = "Human+Neural"
        else:
            suffix_str = suffix.replace("_", "+").upper()
        l = f"BlendRL-CQL ({suffix_str})"
    elif "blendrl_iql" in l.lower():
        suffix = l.lower().split("blendrl_iql_")[-1]
        sub = suffix.replace("_", "+").upper()
        l = f"BlendRL-IQL ({sub})"

    l = l.replace("ppo_cp_tuned", "PPO")
    l = l.replace("ppo_tuned", "PPO")
    l = l.replace("ppo_final_cp", "PPO")
    l = l.replace("blendrl_cp_tuned", "BlendRL")
    l = l.replace("iql_cp_tuned", "IQL")
    l = l.replace("blendrl_iql_cp_tuned", "BlendRL-IQL")
    return l

def get_style_info(label: str) -> Tuple[Optional[str], str, str]:
    l = label.lower()
    if re.search(r'_v\d+', l) or "tune" in l: 
        return None, "-", "o"
    if "ppo" in l and "(on" not in l: return "black", "--", "o"
    if "blendrl-iql" in l: return "#d62728", "-", "s"
    if "fyd" in l: 
        if "human" in l: return "#9467bd", "--", "p"
        return "#9467bd", "-", "p"
    if "cew" in l: 
        if "human" in l: return "#ff7f0e", "--", "x"
        return "#ff7f0e", "-", "h"
    if "blendrl" in l and "iql" not in l:
        if "human" in l: return "#2ca02c", "--", "^"
        return "#2ca02c", "-", "^"
    if "iql" in l and "blendrl" not in l: return "#1f77b4", "-", "d"
    return None, "-", "o"

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

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        raise NotImplementedError("Subclasses must implement run()")
