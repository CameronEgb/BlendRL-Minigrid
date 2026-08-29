"""Slurm training phase runner.

Generates and submits online, offline, and consolidated training jobs to Slurm cluster.
"""
import shlex
from pathlib import Path

from src.pipeline.config import normalize_agent_name
from src.pipeline.datasets import ensure_online_dataset_path, fast_purge_dir, resolve_dataset_path
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
    fast_purge_dir(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Hard-overwrite checkpoints and logs on re-run unless recover=true is explicitly set
    if not cfg.get("recover", False):
        ckpt_dir = Path("results/checkpoints") / cfg.group / cfg.experiment_id
        fast_purge_dir(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        exp_log_dir = Path("results/logs") / cfg.group / cfg.experiment_id
        fast_purge_dir(exp_log_dir)
        exp_log_dir.mkdir(parents=True, exist_ok=True)

    resources = cfg.get("resources", {})
    should_consolidate = cfg.get("consolidate", False)

    # Consolidated Single Slurm Job Execution
    if should_consolidate:
        print(f"\n=== Preparing Consolidated Slurm Job ({cfg.experiment_id}) ===")
        job_name = f"all_{cfg.experiment_id}"
        script_content = generate_sbatch_header(
            job_name=job_name,
            log_dir=log_dir,
            cfg=cfg,
            is_consolidated=True
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
            script_content += f'echo "=== [Phase: Offline Training] Launching all {total_runs} models concurrently ==="\n\n'
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
                script_content += 'echo "Waiting for all concurrent training runs to complete..."\nwait\n\n'

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

        print(f"Submitting Consolidated Slurm Job: {slurm_file}")
        job_id = submit_sbatch(script_content)
        job_ids = [job_id] if job_id else []
        return job_ids, [], True

    job_ids = []
    online_job_ids = {}
    eval_commands = []

    total_online = len(online_list) if not cfg.get("no_online", False) else 0
    total_offline = (len(offline_list) * len(dataset_list)) if not cfg.get("no_offline", False) else 0
    total_jobs_count = total_online + total_offline
    submitted_idx = 0

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
                print(f"Dataset already exists at {dataset_path}. Skipping online training.", flush=True)
                online_job_ids[agent_config] = None
                continue

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
                job_name, overrides_slurm, log_dir=str(log_dir), cfg=cfg, is_consolidated=False
            )
            submitted_idx += 1
            print(f"[{submitted_idx}/{total_jobs_count}] Submitting Online [{agent_config}] ...", end="", flush=True)
            job_id = submit_sbatch(script_content)
            if job_id:
                job_ids.append(job_id)
                online_job_ids[agent_config] = job_id
    else:
        print("\n=== Skipping Online Training Phase ===", flush=True)

    # 2. Offline Training Phases (1 Slurm Job per Method-Dataset Pair -> 1 Process per Node)
    eval_job_ids = []
    if not cfg.get("no_offline", False):
        for agent_config in offline_list:
            agent_name_internal = normalize_agent_name(agent_config)

            for dataset_id in dataset_list:
                dataset_name_internal = normalize_agent_name(dataset_id)
                target_agent_name = f"{agent_name_internal}_{dataset_name_internal}" if len(dataset_list) > 1 else agent_name_internal
                job_name = f"{target_agent_name}_{cfg.experiment_id}"

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

                # Aggregate dependencies if dataset was generated from an online job
                dep_job_id = online_job_ids.get(dataset_id)
                dependency_str = dep_job_id if dep_job_id else None

                script_content = generate_sbatch_header(
                    job_name=job_name,
                    log_dir=log_dir,
                    cfg=cfg,
                    dependency=dependency_str,
                    is_consolidated=False,
                )
                script_content += "\n" + get_shell_env_block(site_cfg) + "\n"
                python_cmd = get_shell_python_cmd(site_cfg)

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
                phase_label = "Offline Sweep" if is_sweep else "Offline Training"
                script_content += f'echo "=== [Phase: {phase_label}] {agent_config} on {dataset_id} ==="\n'
                script_content += f"{python_cmd} src/train.py {train_cmd}\n\n"

                if is_sweep:
                    storage_arg = storage_url if storage_url else ""
                    study_name = f"{cfg.experiment_id}_{target_agent_name}"
                    script_content += f'echo "=== Promoting Winning Checkpoint for {target_agent_name} ==="\n'
                    script_content += f"{python_cmd} -c \"from src.pipeline.optuna_utils import promote_best_trial_checkpoint; promote_best_trial_checkpoint('{cfg.group}', '{cfg.experiment_id}', '{target_agent_name}', '{storage_arg}', '{study_name}')\"\n\n"

                slurm_file = log_dir / f"{job_name}.slurm"
                with open(slurm_file, "w") as f:
                    f.write(script_content)

                submitted_idx += 1
                print(f"[{submitted_idx}/{total_jobs_count}] Submitting Offline [{agent_config}] on [{dataset_id}] ...", end="", flush=True)
                job_id = submit_sbatch(script_content)
                if job_id:
                    job_ids.append(job_id)
    else:
        print("\n=== Skipping Offline Training Phase ===", flush=True)

    # 3. Downstream Plotting Job (dependent on all training jobs)
    if not cfg.get("no_plot", False) and job_ids:
        plot_job_name = f"plot_{cfg.experiment_id}"
        dependency_str = ":".join(job_ids)
        plot_header = generate_sbatch_header(
            job_name=plot_job_name,
            log_dir=log_dir,
            cfg=cfg,
            dependency=dependency_str,
        )
        python_cmd = get_shell_python_cmd(site_cfg)
        plot_cmd = f"{python_cmd} plot/manager.py {cfg.experiment_id}"
        if cfg.get("plot_style", None):
            plot_cmd += f" --style {cfg.get('plot_style', None)}"
        plot_content = plot_header + "\n" + get_shell_env_block(site_cfg) + f"\n\necho \"=== [Generating Final Plots] ===\"\n{plot_cmd}\n"
        plot_slurm_file = log_dir / f"{plot_job_name}.slurm"
        with open(plot_slurm_file, "w") as f:
            f.write(plot_content)
        print(f"[Post-Process] Submitting Plotting Job (dependent on {len(job_ids)} jobs) ...", end="", flush=True)
        submit_sbatch(plot_content)

    job_ids.extend(eval_job_ids)
    return job_ids, eval_commands, False
