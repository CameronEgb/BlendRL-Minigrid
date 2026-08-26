"""Local training phase runner.

Executes online and offline RL training phases sequentially as local subprocesses.
"""
import os
import shutil
import sys
from pathlib import Path

from src.pipeline.config import normalize_agent_name
from src.pipeline.datasets import ensure_online_dataset_path, resolve_dataset_path, run_experiment
from src.pipeline.optuna_utils import (
    create_optuna_study, delete_optuna_study, get_next_study_name, promote_best_trial_checkpoint
)


def run_local_training(cfg, args, online_list, offline_list, dataset_list, sanitized_extra_args, storage_url, is_sweep):
    """Execute online and offline training phases locally as blocking subprocesses."""
    best_online_trial_ids = {}

    ckpt_dir = Path("results/checkpoints") / cfg.group / cfg.experiment_id
    if ckpt_dir.exists():
        print(f"Clearing old checkpoints in {ckpt_dir} for fresh experiment execution...")
        try:
            shutil.rmtree(ckpt_dir)
        except OSError as e:
            print(f"Notice: Could not clear checkpoint dir {ckpt_dir}: {e}")

    # 1. Online Training Phases
    if not args.no_online:
        for agent_config in online_list:
            agent_name_internal = normalize_agent_name(agent_config)
            study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name_internal)
            
            dataset_path, has_pkl = ensure_online_dataset_path(
                group=cfg.group,
                experiment_id=cfg.experiment_id,
                agent_name_internal=agent_name_internal,
                is_sweep=is_sweep
            )

            if has_pkl:
                print(f"Dataset already exists at {dataset_path}. Skipping online training.")
                best_online_trial_ids[agent_config] = "0"
                continue

            print(f"\n=== Phase: Online Training ({agent_config}) ===")
            overrides = [
                f"+experiment={args.experiment}",
                f"++local=true",
                f"mode=online",
                f"agent={agent_config}",
                f"++agent.name={agent_name_internal}",
                f"++dataset_path={dataset_path}",
                f"++hydra.sweeper.study_name={study_name}"
            ] + sanitized_extra_args
            
            if is_sweep:
                delete_optuna_study(storage_url, study_name)
                direction = cfg.hydra.sweeper.get("direction", "maximize") if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper") else "maximize"
                create_optuna_study(storage_url, study_name, direction=direction)
                
            run_experiment(overrides)
            
            # After training, find the best trial ID if we were sweeping
            if is_sweep:
                best_id = promote_best_trial_checkpoint(cfg.group, cfg.experiment_id, agent_name_internal, storage_url, study_name)
                best_online_trial_ids[agent_config] = best_id
            else:
                best_online_trial_ids[agent_config] = "0"
    else:
        print("\n=== Skipping Online Training Phase ===")

    # 2. Offline Training Phases (Many-to-Many)
    if not args.no_offline:
        for dataset_id in dataset_list:
            dataset_name_internal = normalize_agent_name(dataset_id)
            
            best_id = best_online_trial_ids.get(dataset_id, "0")
            best_trial_path = Path("in/datasets") / cfg.group / cfg.experiment_id / dataset_name_internal / best_id
            yaml_ds_path = cfg.mode.get("dataset_path", None) if hasattr(cfg, "mode") else None
            if best_trial_path.exists() and any(best_trial_path.glob("*.pkl")):
                dataset_path = best_trial_path
            else:
                try:
                    dataset_path = resolve_dataset_path(dataset_id, group=cfg.group, experiment_id=cfg.experiment_id, yaml_ds_path=yaml_ds_path)
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    sys.exit(1)
            print(f"Using dataset from: {dataset_path}")

            for agent_config in offline_list:
                agent_name_internal = normalize_agent_name(agent_config)
                study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name_internal)
                
                print(f"\n=== Phase: Offline Training ({agent_config}) on Dataset ({dataset_id}) ===")
                dataset_path_override = any("mode.dataset_path=" in arg for arg in sanitized_extra_args)
                target_agent_name = f"{agent_name_internal}_{dataset_name_internal}" if len(dataset_list) > 1 else agent_name_internal
                overrides = [
                    f"+experiment={args.experiment}",
                    f"++local=true",
                    f"mode=offline",
                    f"agent={agent_config}",
                    f"++agent.name={target_agent_name}",
                    f"++hydra.sweeper.study_name={study_name}"
                ]
                if cfg.env.name == "pyrenees":
                    ruleset = "default" if dataset_id == "problem" else "step"
                    overrides.append(f"++env.rules={ruleset}")
                    overrides.append(f"++env.problem_type={dataset_id}")
                if not dataset_path_override:
                    overrides.append(f"++mode.dataset_path={dataset_path}")
                overrides += sanitized_extra_args
                
                if is_sweep:
                    delete_optuna_study(storage_url, study_name)
                    direction = cfg.hydra.sweeper.get("direction", "minimize") if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper") else "minimize"
                    create_optuna_study(storage_url, study_name, direction=direction)
                    
                run_experiment(overrides)

                if is_sweep:
                    promote_best_trial_checkpoint(cfg.group, cfg.experiment_id, agent_name_internal, storage_url, study_name)
    else:
        print("\n=== Skipping Offline Training Phase ===")
