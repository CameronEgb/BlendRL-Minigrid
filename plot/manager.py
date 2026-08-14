import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import importlib
import pkgutil
import shutil
import yaml

from plot.base import BasePlotter

# Environment-based default plot modules.
# Experiments can override by defining a `plots:` section in their YAML.
ENV_DEFAULT_PLOTS = {
    "mimic":    ["losses", "policy_eval", "reports"],
    "cartpole": ["convergence", "losses", "reports"],
}
FALLBACK_PLOTS = ["convergence", "losses", "reports"]


def discover_plotters() -> dict:
    """Auto-discover all BasePlotter subclasses in the plot/ package."""
    registry = {}
    plot_pkg_dir = str(Path(__file__).parent)
    for importer, modname, ispkg in pkgutil.iter_modules([plot_pkg_dir]):
        if modname.startswith("_") or modname in ("base", "manager"):
            continue
        try:
            mod = importlib.import_module(f"plot.{modname}")
            for attr_name in dir(mod):
                cls = getattr(mod, attr_name)
                if (isinstance(cls, type) and issubclass(cls, BasePlotter) 
                        and cls is not BasePlotter and hasattr(cls, 'name')):
                    inst = cls()
                    registry[inst.name] = cls
        except Exception as e:
            print(f"Warning: Could not load plotter module 'plot.{modname}': {e}")
    return registry


def get_requested_plots(exp_cfg: dict, exp_id: str) -> dict:
    """Determine which plots to generate based on experiment config.
    
    Priority:
      1. Explicit `plots:` section in experiment YAML (list or dict form)
      2. Environment-based defaults from ENV_DEFAULT_PLOTS
      3. FALLBACK_PLOTS
    """
    plots_val = exp_cfg.get("plots", None)
    if plots_val is not None:
        if isinstance(plots_val, list):
            return {name: {} for name in plots_val}
        elif isinstance(plots_val, dict):
            return plots_val

    # Fall back to env-based defaults
    env_name = ""
    env_val = exp_cfg.get("env", {})
    if isinstance(env_val, dict):
        env_name = env_val.get("name", "")
    elif isinstance(env_val, str):
        env_name = env_val

    if not env_name and "group" in exp_cfg:
        env_name = exp_cfg["group"]
    if not env_name and "/" in exp_id:
        env_name = exp_id.split("/")[0]

    # Also check for early_prediction task types
    task_name = exp_cfg.get("task", "")
    defaults = ENV_DEFAULT_PLOTS.get(env_name, FALLBACK_PLOTS)
    
    # Auto-include early_prediction if relevant
    result = {name: {} for name in defaults}
    if "early_prediction" not in result:
        if "early_pred" in exp_id or "early_prediction" in task_name:
            result["early_prediction"] = {}
    
    return result


def run_experiment_plots(exp_id: str, exp_config_name: str = None, style: str = None):
    print(f"\n==================================================")
    print(f"=== Auto-Generating Plots for Experiment: {exp_id} ===")
    print(f"==================================================")

    dummy_plotter = BasePlotter("manager")
    exp_cfg = dummy_plotter.get_experiment_config(exp_id, exp_config_name=exp_config_name)
    group = dummy_plotter.get_group(exp_id, exp_cfg)
    clean_exp = Path(exp_id).stem
    output_dir = Path("results/plots") / group / clean_exp

    if output_dir.exists():
        print(f"Clearing previous plots from: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = discover_plotters()
    requested_modules = get_requested_plots(exp_cfg, exp_id)

    for module_name, overrides in requested_modules.items():
        if module_name in registry:
            plotter_cls = registry[module_name]
            plotter = plotter_cls()
            try:
                plot_overrides = overrides if isinstance(overrides, dict) else {}
                if style:
                    plot_overrides["style"] = style
                plotter.run(exp_id, cli_overrides=plot_overrides if plot_overrides else None)
            except Exception as e:
                print(f"Error running plotter '{module_name}' for '{exp_id}': {e}")
        else:
            print(f"Warning: Unknown plotter module '{module_name}' requested. Available: {list(registry.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate Experiment Plot Generation")
    parser.add_argument("experiment_id", type=str, help="Experiment ID to generate plots for")
    parser.add_argument("--experiment", "-c", "--config", dest="experiment", type=str, default=None, help="Base experiment config name if different from experiment_id")
    parser.add_argument("--style", type=str, default=None, help="Plot style config")
    args = parser.parse_args()

    run_experiment_plots(args.experiment_id, exp_config_name=args.experiment, style=args.style)
