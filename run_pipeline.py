"""Unified experiment pipeline orchestrator for NeSyRL.

Coordinates online and offline RL methods, Optuna sweeps, and Slurm cluster submissions.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import hydra
from hydra import compose, initialize

from src.method_registry import clean_label
from src.pipeline.config import (
    normalize_agent_name, parse_method_list, resolve_experiment_config_name
)
from src.pipeline.datasets import run_plotting
from src.pipeline.local_runner import run_local_training
from src.pipeline.optuna_utils import launch_optuna_dashboard
from src.pipeline.slurm import generate_sbatch_header, submit_sbatch
from src.pipeline.slurm_runner import run_slurm_training
from src.pipeline.validation import validate_experiment_config


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python run_pipeline.py <group>/<experiment_id> [Hydra Overrides]")
        print("Standard Orchestration Overrides (use Hydra syntax, e.g. key=value):")
        print("  site=local            Interactive CLI execution (default, for local machine or cluster interactive node)")
        print("  site=ncshare          Slurm cluster execution on NCShare")
        print("  site=arc              Slurm cluster execution on ARC")
        print("  no_plot=true          Skip automatic plotting")
        print("  no_online=true        Skip online training phase")
        print("  no_offline=true       Skip offline training phase")
        print("  dry_run=true          Validate config and exit")
        print("  sweep=true            Run Optuna hyperparameter sweep")
        print("  dash=true             Launch Optuna dashboard during run")
        print("  dash_only=true        Launch Optuna dashboard and exit")
        print("  remake=true           Force recalculation of summaries")
        print("  consolidate=true      Consolidate Slurm jobs into 1 single GPU job")
        print("  experiment_id=...     Override the experiment directory name")
        sys.exit(0)

    raw_experiment = sys.argv[1]
    extra_args = sys.argv[2:]

    # Resolve experiment config path
    experiment_arg = resolve_experiment_config_name(raw_experiment)

    # Prepare pure Hydra compose overrides (filtering out sweep logic that crashes compose API)
    overrides_for_compose = [f"+experiment={experiment_arg}", f"++experiment_name={experiment_arg}"]
    for arg in extra_args:
        # The compose API cannot parse Optuna sweep operators or hydra internal configs
        if any(sw in arg for sw in ["interval(", "choice(", "range(", "hydra.", "hydra/"]):
            continue
        if "=" in arg:
            overrides_for_compose.append(arg)
            
    # Subprocesses need everything PLUS the internal experiment tracking
    sanitized_extra_args = list(extra_args)
    sanitized_extra_args.append(f"++experiment_name={experiment_arg}")

    # Load configuration
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="in/config")
        cfg = compose(config_name="config", overrides=overrides_for_compose, return_hydra_config=True)
        exp_stem = Path(experiment_arg).stem
        if not cfg.get("experiment_id") or cfg.experiment_id == "default_exp":
            cfg.experiment_id = exp_stem
            
        exp_group = Path(experiment_arg).parent.name if "/" in experiment_arg else "ungrouped"
        if not cfg.get("group") or cfg.group == "ungrouped":
            cfg.group = exp_group
            
        # Ensure it's in extra args so it passes to children
        if not any("experiment_id=" in arg for arg in sanitized_extra_args):
            sanitized_extra_args.append(f"++experiment_id={cfg.experiment_id}")
        if not any("group=" in arg for arg in sanitized_extra_args):
            sanitized_extra_args.append(f"++group={cfg.group}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    is_sweep = cfg.get("sweep", False) or "--multirun" in sanitized_extra_args or "-m" in sanitized_extra_args
    if is_sweep and "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
        sanitized_extra_args.append("--multirun")

    # Pre-flight validation
    try:
        notices = validate_experiment_config(cfg, experiment_arg, is_sweep=is_sweep)
        for n in notices:
            print(f"[Config Notice] {n}")
    except ValueError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    if cfg.get("dry_run", False):
        print(f"\n[Validation Success] Experiment config '{experiment_arg}' is valid and ready to run.")
        sys.exit(0)

    # Execution mode is determined solely by the site profile: site=local -> interactive CLI, any other site -> Slurm cluster
    site_name = getattr(cfg.site, "name", "local") if hasattr(cfg, "site") else "local"
    is_interactive = (site_name == "local")
    print(f"Execution Mode: {'Interactive (Local CLI)' if is_interactive else f'Slurm Cluster ({site_name})'}")

    storage_url = None
    if "hydra" in cfg and "sweeper" in cfg.hydra and "storage" in cfg.hydra.sweeper:
        storage_url = cfg.hydra.sweeper.storage
        if storage_url:
            storage_url = str(storage_url).replace("${experiment_id}", cfg.experiment_id)
        import os
        os.makedirs("results/optuna", exist_ok=True)
        
    if is_interactive and storage_url and (cfg.get("dash") or cfg.get("dash_only")):
        launch_optuna_dashboard(storage_url)
        if cfg.get("dash_only"):
            print("Dashboard running in persistent mode. Press Ctrl+C to exit.")
            import time
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                sys.exit(0)
                
    if cfg.get("dash_only"):
        print("Error: Could not find Optuna storage URL in configuration.")
        sys.exit(1)

    online_methods = cfg.get("online_methods", "")
    offline_methods = cfg.get("offline_methods", "")
    offline_datasets = cfg.get("offline_datasets", "")

    online_list = parse_method_list(online_methods)
    offline_list = parse_method_list(offline_methods)
    dataset_list = parse_method_list(offline_datasets)

    print(f"Detected Online Methods: {online_list}")
    print(f"Detected Offline Methods: {offline_list}")
    print(f"Using Datasets for Offline Training: {dataset_list}")

    # Build context for tasks
    context = {
        "is_interactive": is_interactive,
        "site_name": site_name,
        "local_val": is_interactive,
        "sanitized_extra_args": sanitized_extra_args,
        "storage_url": storage_url,
        "is_sweep": is_sweep,
        "online_list": online_list,
        "offline_list": offline_list,
        "dataset_list": dataset_list,
    }

    # Extract task name
    task_name = cfg.get("task", "rl")
    if not task_name:
        task_name = "rl"

    from src.pipeline.task_registry import get_task, register_task

    @register_task("rl")
    def run_standard_rl_task(cfg, context):
        if context["is_interactive"]:
            run_local_training(cfg, context)
        else:
            run_slurm_training(cfg, context)

    
    task_fn = get_task(task_name)
    
    # Introspect task_fn to see if it accepts args (backwards compatibility for custom tasks)
    import inspect
    sig = inspect.signature(task_fn)
    if "args" in sig.parameters:
        task_fn(cfg, None, context)
    else:
        task_fn(cfg, context)

if __name__ == "__main__":
    main()
