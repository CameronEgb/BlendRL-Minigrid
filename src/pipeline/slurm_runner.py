"""Slurm training phase runner.

Generates and submits online, offline, and consolidated training jobs to Slurm cluster.
"""
import shlex
from pathlib import Path

from src.pipeline.config import normalize_agent_name
from src.pipeline.datasets import ensure_online_dataset_path, resolve_dataset_path
from src.pipeline.optuna_utils import create_optuna_study, delete_optuna_study, get_next_study_name
from src.pipeline.slurm import generate_sbatch_header, generate_sbatch_script, submit_sbatch
from src.pipeline.commands import build_online_overrides, build_offline_overrides, get_sweep_direction
from src.pipeline.runtime import get_shell_env_block, get_shell_python_cmd


def run_slurm_training(cfg, context):
    online_list = context["online_list"]
    offline_list = context["offline_list"]
    dataset_list = context["dataset_list"]
    sanitized_extra_args = context["sanitized_extra_args"]
    storage_url = context["storage_url"]
    is_sweep = context["is_sweep"]
    site_cfg = cfg.site

    """Submit online and offline training jobs to Slurm cluster.
    
    Returns:
        tuple[list[str], list[str], bool]: (job_ids, eval_commands, is_consolidated)
    """
    log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
    if log_dir.exists():
        print(f"Clearing old logs in {log_dir}...")
        for log_file in log_dir.glob("*"):
            if log_file.is_file():
                log_file.unlink()
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path("results/checkpoints") / cfg.group / cfg.experiment_id
    if ckpt_dir.exists():
        import time
        backup_path = f"{ckpt_dir}_backup_{int(time.time())}"
        print(f"Backing up old checkpoints from {ckpt_dir} to {backup_path}...")
        ckpt_dir.rename(backup_path)

    resources = cfg.get("resources", {})
    should_consolidate = cfg.get("consolidate", False)

    # Consolidated Single 1-GPU Slurm Job Execution
    if should_consolidate:
        print(f"\n=== Preparing Consolidated 1-GPU Slurm Job ({cfg.experiment_id}) ===")
        job_name = f"all_{cfg.experiment_id}"
        script_content = generate_sbatch_header(
            job_name=job_name,
            log_dir=log_dir,
            cfg=cfg
        )
        script_content += "\n" + get_shell_env_block(site_cfg) + "\n"
        python_cmd = get_shell_python_cmd(site_cfg)

        # 1. Online Training Commands
        if not cfg.get("no_online", False):
            for agent_config in online_list:
                agent_name_internal = normalize_agent_name(agent_config)
                dataset_path = f"in/datasets/{cfg.group}/{cfg.experiment_id}/{agent_name_internal}"
                cmd_args = build_online_overrides(
                    experiment=cfg.get("experiment_name", ""),
                    agent_config=agent_config,
                    agent_name=agent_name_internal,
                    dataset_path=dataset_path,
                    local_val=False,
                    extra_args=sanitized_extra_args
                )
                if is_sweep:
                    study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name_internal)
                    direction = get_sweep_direction(cfg, "online")
                    create_optuna_study(storage_url, study_name, direction=direction)
                    cmd_args.append(f"++hydra.sweeper.study_name={study_name}")
                train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                script_content += f'echo "=== [Phase: Online Training] {agent_config} ==="\n'
                script_content += f"{python_cmd} src/train.py {train_cmd}\n\n"

        # 2. Offline Training Commands (All methods & datasets in parallel)
        if not cfg.get("no_offline", False):
            total_runs = len(offline_list) * len(dataset_list)
            script_content += f'echo "=== [Phase: Offline Training] Launching all {total_runs} models concurrently on GPU ==="\n\n'
            for agent_config in offline_list:
                agent_name_internal = normalize_agent_name(agent_config)
                for dataset_id in dataset_list:
                    dataset_name_internal = normalize_agent_name(dataset_id)
                    yaml_ds_path = cfg.mode.get("dataset_path", None) if hasattr(cfg, "mode") else None
                    try:
                        dataset_path = resolve_dataset_path(
                            dataset_id=dataset_name_internal,
                            group=cfg.group,
                            experiment_id=cfg.experiment_id,
                            yaml_ds_path=yaml_ds_path
                        )
                    except FileNotFoundError:
                        dataset_path = Path("in/datasets") / cfg.group / dataset_name_internal

                    target_agent_name = f"{agent_name_internal}_{dataset_name_internal}" if len(dataset_list) > 1 else agent_name_internal
                    cmd_args = build_offline_overrides(
                        experiment=cfg.get("experiment_name", ""),
                        agent_config=agent_config,
                        agent_name=target_agent_name,
                        dataset_path=str(dataset_path),
                        local_val=False,
                        dataset_id=dataset_id,
                        extra_args=sanitized_extra_args
                    )
                    if is_sweep:
                        study_name = get_next_study_name(cfg.group, cfg.experiment_id, target_agent_name)
                        direction = get_sweep_direction(cfg, "offline")
                        create_optuna_study(storage_url, study_name, direction=direction)
                        cmd_args.append(f"++hydra.sweeper.study_name={study_name}")
                        if "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
                            cmd_args.append("--multirun")
                    train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                    if total_runs > 1:
                        script_content += f'echo "  -> Starting [{agent_config}] on [{dataset_id}] in background..."\n'
                        script_content += f"{python_cmd} src/train.py {train_cmd} &\n\n"
                    else:
                        script_content += f"{python_cmd} src/train.py {train_cmd}\n\n"

            if total_runs > 1:
                script_content += 'echo "Waiting for all concurrent training runs to complete on GPU..."\nwait\n\n'

        if is_sweep:
            script_content += 'echo "=== Promoting Winning Checkpoints for All Methods & Datasets ==="\n'
            storage_arg = storage_url if storage_url else ""
            for dataset_id in dataset_list:
                dataset_name_internal = normalize_agent_name(dataset_id)
                for agent_config in offline_list:
                    agent_name_internal = normalize_agent_name(agent_config)
                    target_agent_name = f"{agent_name_internal}_{dataset_name_internal}" if len(dataset_list) > 1 else agent_name_internal
                    study_name = f"{cfg.experiment_id}_{target_agent_name}"
                    script_content += f"{python_cmd} -c \"from src.pipeline.optuna_utils import promote_best_trial_checkpoint; promote_best_trial_checkpoint('{cfg.group}', '{cfg.experiment_id}', '{target_agent_name}', '{storage_arg}', '{study_name}')\"\n"
            script_content += '\n'

        # 3. Final Plotting
        if not cfg.get("no_plot", False):
            plot_cmd = f"{python_cmd} plot/manager.py {cfg.experiment_id}"
            if cfg.get("plot_style", None):
                plot_cmd += f" --style {cfg.get('plot_style', None)}"
            script_content += f'echo "=== [Generating Final Plots] ==="\n'
            script_content += f"{plot_cmd}\n\n"

        slurm_file = log_dir / f"consolidated_{cfg.experiment_id}.slurm"
        with open(slurm_file, "w") as f:
            f.write(script_content)

        print(f"Submitting Consolidated 1-GPU Slurm Job: {slurm_file}")
        job_id = submit_sbatch(script_content)
        job_ids = [job_id] if job_id else []
        return job_ids, [], True

    job_ids = []
    online_job_ids = {}
    eval_commands = []

    # 1. Online Training Phases
    if not cfg.get("no_online", False):
        for agent_config in online_list:
            agent_name_internal = normalize_agent_name(agent_config)
            
            dataset_path, has_pkl = ensure_online_dataset_path(
                group=cfg.group,
                experiment_id=cfg.experiment_id,
                agent_name_internal=agent_name_internal,
                is_sweep=is_sweep
            )

            if has_pkl:
                print(f"Dataset already exists at {dataset_path}. Skipping online training.")
                online_job_ids[agent_config] = None
                continue

            print(f"\n=== Preparing Slurm Job: Online Training ({agent_config}) ===")
            job_name = f"{agent_name_internal}_{cfg.experiment_id}"
            
            dataset_arg = str(dataset_path) if not is_sweep else None
            overrides_slurm = build_online_overrides(
                experiment=cfg.get("experiment_name", ""),
                agent_config=agent_config,
                agent_name=agent_name_internal,
                dataset_path=dataset_arg,
                local_val=False,
                extra_args=sanitized_extra_args
            )
            
            if is_sweep:
                study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name_internal)
                direction = get_sweep_direction(cfg, "online")
                create_optuna_study(storage_url, study_name, direction=direction)
                overrides_slurm.append(f"++hydra.sweeper.study_name={study_name}")
            
            script_content = generate_sbatch_script(
                job_name, overrides_slurm, log_dir=str(log_dir), cfg=cfg
            )
            job_id = submit_sbatch(script_content)
            if job_id:
                job_ids.append(job_id)
                online_job_ids[agent_config] = job_id
    else:
        print("\n=== Skipping Online Training Phase ===")

    # 2. Offline Training Phases (1 Slurm Job per Method, running datasets in parallel on GPU)
    eval_job_ids = []
    if not cfg.get("no_offline", False):
        for agent_config in offline_list:
            agent_name_internal = normalize_agent_name(agent_config)
            job_name = f"{agent_name_internal}_{cfg.experiment_id}"
            
            # Aggregate dependencies if datasets were generated from online jobs
            deps = [online_job_ids.get(ds) for ds in dataset_list if online_job_ids.get(ds)]
            dependency_str = ":".join(deps) if deps else None

            script_content = generate_sbatch_header(
                job_name=job_name,
                log_dir=log_dir,
                cfg=cfg,
                dependency=dependency_str,
            )
            script_content += "\n" + get_shell_env_block(site_cfg) + "\n"
            python_cmd = get_shell_python_cmd(site_cfg)

            for dataset_id in dataset_list:
                dataset_name_internal = normalize_agent_name(dataset_id)
                yaml_ds_path = cfg.mode.get("dataset_path", None) if hasattr(cfg, "mode") else None
                try:
                    dataset_path = resolve_dataset_path(
                        dataset_id=dataset_name_internal,
                        group=cfg.group,
                        experiment_id=cfg.experiment_id,
                        yaml_ds_path=yaml_ds_path
                    )
                except FileNotFoundError:
                    dataset_path = Path("in/datasets") / cfg.group / dataset_name_internal

                target_agent_name = f"{agent_name_internal}_{dataset_name_internal}" if len(dataset_list) > 1 else agent_name_internal
                cmd_args = build_offline_overrides(
                    experiment=cfg.get("experiment_name", ""),
                    agent_config=agent_config,
                    agent_name=target_agent_name,
                    dataset_path=str(dataset_path),
                    local_val=False,
                    dataset_id=dataset_id,
                    extra_args=sanitized_extra_args
                )
                
                if is_sweep:
                    study_name = get_next_study_name(cfg.group, cfg.experiment_id, target_agent_name)
                    direction = get_sweep_direction(cfg, "offline")
                    create_optuna_study(storage_url, study_name, direction=direction)
                    cmd_args.append(f"++hydra.sweeper.study_name={study_name}")
                    if "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
                        cmd_args.append("--multirun")
                
                train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                
                if len(dataset_list) > 1:
                    phase_label = "Offline Sweep (Parallel GPU)" if is_sweep else "Offline Training (Parallel GPU)"
                    script_content += f'echo "=== [Phase: {phase_label}] {agent_config} on {dataset_id} ==="\n'
                    script_content += f"{python_cmd} src/train.py {train_cmd} &\n\n"
                else:
                    phase_label = "Offline Sweep" if is_sweep else "Offline Training"
                    script_content += f'echo "=== [Phase: {phase_label}] {agent_config} on {dataset_id} ==="\n'
                    script_content += f"{python_cmd} src/train.py {train_cmd}\n\n"

            if len(dataset_list) > 1:
                script_content += 'echo "Waiting for all concurrent problem sweeps to complete on GPU..."\nwait\n\n'

            if is_sweep:
                script_content += 'echo "=== Promoting Winning Checkpoints for All Datasets ==="\n'
                storage_arg = storage_url if storage_url else ""
                for d_id in dataset_list:
                    d_name = normalize_agent_name(d_id)
                    t_agent = f"{agent_name_internal}_{d_name}" if len(dataset_list) > 1 else agent_name_internal
                    s_name = f"{cfg.experiment_id}_{t_agent}"
                    script_content += f"{python_cmd} -c \"from src.pipeline.optuna_utils import promote_best_trial_checkpoint; promote_best_trial_checkpoint('{cfg.group}', '{cfg.experiment_id}', '{t_agent}', '{storage_arg}', '{s_name}')\"\n"
                script_content += '\n'

            slurm_file = log_dir / f"{job_name}.slurm"
            with open(slurm_file, "w") as f:
                f.write(script_content)

            print(f"\nSubmitting Method Slurm Job ({agent_config} -> {len(dataset_list)} datasets in parallel on GPU): {slurm_file}")
            job_id = submit_sbatch(script_content)
            if job_id:
                job_ids.append(job_id)
    else:
        print("\n=== Skipping Offline Training Phase ===")

    job_ids.extend(eval_job_ids)
    return job_ids, eval_commands, False
