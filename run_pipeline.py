"""Unified experiment pipeline orchestrator for NeSyRL.

Coordinates online and offline RL methods, Optuna sweeps, and Slurm cluster submissions.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
from hydra import compose, initialize

from src.method_registry import clean_label
from src.pipeline.config import (
    filter_pipeline_args, normalize_agent_name, parse_method_list, resolve_experiment_config_name
)
from src.pipeline.datasets import run_early_prediction_eval, run_plotting
from src.pipeline.early_prediction_task import run_early_prediction_task
from src.pipeline.reciprocal_task import run_reciprocal_refinement
from src.pipeline.local_runner import run_local_training
from src.pipeline.optuna_utils import (
    DEFAULT_OPTUNA_DB_URL, get_best_trial_id, launch_optuna_dashboard
)
from src.pipeline.slurm import generate_sbatch_header, submit_sbatch
from src.pipeline.slurm_runner import run_slurm_training
from src.pipeline.validation import validate_experiment_config


def main():
    parser = argparse.ArgumentParser(description="NeSyRL Unified Experiment Pipeline")
    parser.add_argument("experiment", type=str, help="Experiment name from in/config/experiment/")
    parser.add_argument("-e", "--exp-id", "--experiment-id", dest="exp_id", type=str, default=None, help="Override experiment ID (run/output directory name to prevent overwriting)")
    parser.add_argument("--local", type=str, default=None, help="Force local run (true/false)")
    parser.add_argument("--partition", type=str, default="gpu", help="Slurm partition")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per job")
    parser.add_argument("--cores", type=int, default=16, help="Number of CPU cores per job")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes per job")
    parser.add_argument("--time", type=str, default="01:00:00", help="Slurm walltime limit per job (default: 01:00:00)")
    parser.add_argument("--consolidate", action="store_true", default=False, help="Consolidate all training methods, evals, and plots into 1 single GPU Slurm job")
    parser.add_argument("--plot-style", type=str, default=None, help="Style config for plotter")
    parser.add_argument("--no-plot", action="store_true", help="Skip automatic plotting")
    parser.add_argument("--no-online", action="store_true", help="Skip online training phase")
    parser.add_argument("--no-offline", action="store_true", help="Skip offline training phase")
    parser.add_argument("--dry-run", "--validate", dest="dry_run", action="store_true", help="Validate experiment configuration and exit without running")
    parser.add_argument("--dash", action="store_true", help="Launch the Optuna dashboard during the run")
    parser.add_argument("--dash-only", action="store_true", help="Launch the persistent dashboard and exit")
    parser.add_argument("-s", "--sweep", action="store_true", help="Run hyperparameter sweep (enables Hydra --multirun)")
    parser.add_argument("--remake", action="store_true", help="Force recalculation and overwrite of early prediction summaries")
    args, extra_args = parser.parse_known_args()

    # Resolve experiment config path (supporting group subdirectories like mimic/mimic_cql or mimic_cql)
    args.experiment = resolve_experiment_config_name(args.experiment)

    # Sanitize CLI extra arguments and prepare Hydra compose overrides
    sanitized_extra_args, overrides_for_compose = filter_pipeline_args(
        extra_args, experiment=args.experiment, exp_id=args.exp_id
    )

    if args.sweep and "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
        sanitized_extra_args.append("--multirun")

    # Load configuration to get method lists and Optuna info
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="in/config")
        cfg = compose(config_name="config", overrides=overrides_for_compose, return_hydra_config=True)
        exp_stem = Path(args.experiment).stem
        if args.exp_id:
            cfg.experiment_id = args.exp_id
        elif not cfg.get("experiment_id") or cfg.experiment_id == "default_exp":
            cfg.experiment_id = exp_stem
        if not any("experiment_id=" in arg for arg in sanitized_extra_args):
            sanitized_extra_args.append(f"++experiment_id={cfg.experiment_id}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    is_sweep = args.sweep or "--multirun" in sanitized_extra_args or "-m" in sanitized_extra_args

    # Pre-flight validation
    try:
        notices = validate_experiment_config(cfg, args.experiment, is_sweep=is_sweep)
        for n in notices:
            print(f"[Config Notice] {n}")
    except ValueError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    if args.dry_run:
        print(f"\n[Validation Success] Experiment config '{args.experiment}' is valid and ready to run.")
        sys.exit(0)

    # Determine execution mode (local vs slurm cluster)
    local_val = cfg.get("local", True)
    if args.local is not None:
        local_val = args.local.lower() in ("true", "1", "yes")

    print(f"Execution Mode: {'Local' if local_val else 'Slurm Cluster'}")

    # Check for Optuna storage in the config
    storage_url = None
    if "hydra" in cfg and "sweeper" in cfg.hydra and "storage" in cfg.hydra.sweeper:
        storage_url = cfg.hydra.sweeper.storage
        os.makedirs("results/optuna", exist_ok=True)
        
    if local_val and storage_url and (args.dash or args.dash_only):
        storage_url = storage_url.replace("${experiment_id}", cfg.experiment_id)
        launch_optuna_dashboard(storage_url)
        if args.dash_only:
            print("Dashboard running in persistent mode. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                sys.exit(0)
                
    if args.dash_only:
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
    
    # Auto-convert MIMIC dataset if needed
    if cfg.env.name == "mimic":
        target_npz = cfg.env.get("dataset_name", "mimic_lazy_0_interventions_balanced.npz")
        npz_stem = Path(target_npz).stem
        target_dir = Path("in/datasets/mimic") / npz_stem
        cql_dir = Path("in/datasets/mimic/cql")
        if not ((target_dir.exists() and any(target_dir.glob("*.pkl"))) or (cql_dir.exists() and any(cql_dir.glob("*.pkl")))):
            print(f"\n=== Auto-Converting MIMIC NPZ Dataset ({target_npz}) to PKL Format ===")
            subprocess.run([sys.executable, "scripts/convert_npz_to_pkl.py", target_npz], check=True)

    print(f"Using Datasets for Offline Training: {dataset_list}")

    # Standalone Reciprocal Refinement Task (iterative EP ↔ CQL co-training)
    if cfg.get("task", "") == "reciprocal_refinement":
        run_reciprocal_refinement(cfg, args, local_val)
        sys.exit(0)

    # Standalone Early Prediction Tasks
    if cfg.get("task", "").startswith("early_prediction"):
        run_early_prediction_task(cfg, args, local_val, sanitized_extra_args, storage_url, is_sweep)

    # Main RL Pipeline Execution
    job_ids = []
    eval_commands = []
    bundled_plot = False
    if local_val:
        run_local_training(cfg, args, online_list, offline_list, dataset_list, sanitized_extra_args, storage_url, is_sweep)
    else:
        job_ids, eval_commands, bundled_plot = run_slurm_training(cfg, args, online_list, offline_list, dataset_list, sanitized_extra_args, storage_url, is_sweep)

    # Local Early Prediction Evaluation (for early_prediction tasks)
    if local_val and cfg.get("task", "").startswith("early_prediction"):
        print("\n=== Phase: Local Early Prediction Evaluation ===")
        ckpt_dir_root = Path("results/checkpoints") / cfg.group / cfg.experiment_id
        if ckpt_dir_root.exists() and any(ckpt_dir_root.rglob("best_model*.ckpt")):
            run_early_prediction_eval(ckpt_dir_root, remake=args.remake)

    # Plotting & Dependent Job Assembly
    if not args.no_plot:
        if local_val:
            run_plotting(cfg.experiment_id, style=args.plot_style, base_experiment=args.experiment)
        else:
            if not bundled_plot and job_ids:
                actual_exp_id = cfg.experiment_id
                print(f"\n=== Preparing Final Job: Evaluation and Plotting ({actual_exp_id}) ===")
                job_name = f"final_{actual_exp_id}"
                log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
                all_dependencies = ":".join(jid for jid in job_ids if jid != "99999")
                
                project_root = os.getcwd()
                plot_cmd = f"{project_root}/venv/bin/python3 plot/manager.py {actual_exp_id}"
                if args.experiment and args.experiment != actual_exp_id:
                    plot_cmd += f" --experiment {args.experiment}"
                if args.plot_style:
                    plot_cmd += f" --style {args.plot_style}"
                    
                final_script = generate_sbatch_header(
                    job_name=job_name,
                    log_dir=log_dir,
                    partition=args.partition,
                    gpus=0,
                    cores=1,
                    nodes=1,
                    dependency=all_dependencies if all_dependencies else None,
                    dependency_type="afterany"
                )
                final_script += f"\nexport PROJECT_ROOT={project_root}\n"
                final_script += f"export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH\n\n"
                
                if eval_commands:
                    final_script += "echo 'Running combined evaluations...'\n"
                    for cmd in eval_commands:
                        final_script += f"{cmd}\n\n"
                        
                final_script += f"echo 'Generating final plots for {actual_exp_id}...'\n"
                final_script += f"{plot_cmd}\n"
                
                job_id = submit_sbatch(final_script)
                if job_id:
                    job_ids.append(job_id)
                        
            # Save Job IDs to a file for easy cancellation
            ids_file = Path("results/slurm_ids") / f"{cfg.experiment_id}.txt"
            ids_file.parent.mkdir(parents=True, exist_ok=True)
            with open(ids_file, "w") as f:
                for jid in job_ids:
                    if jid != "99999":
                        f.write(f"{jid}\n")
            
            ran_methods = []
            for m in online_list + offline_list:
                cl = clean_label(normalize_agent_name(m))
                if cl not in ran_methods:
                    ran_methods.append(cl)

            print(f"\n" + "="*40)
            print(f"SUBMISSION SUMMARY")
            print(f"Experiment:    {cfg.group}/{cfg.experiment_id}")
            print(f"Partition:     {args.partition}")
            print(f"Methods Ran:")
            for m in ran_methods:
                print(f"  - {m}")
            print(f"To cancel this experiment, run:")
            print(f"  ./scripts/cancel.sh {cfg.experiment_id}")
            print("="*40)

            print(f"\n=== Submission Complete ===")
            sys.stdout.flush()
            sys.exit(0)


if __name__ == "__main__":
    main()
