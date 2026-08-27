"""Slurm training phase runner.

Generates and submits online, offline, and consolidated training jobs to Slurm cluster.
"""
import os
import shlex
import shutil
from pathlib import Path

from src.pipeline.config import normalize_agent_name
from src.pipeline.datasets import ensure_online_dataset_path, resolve_dataset_path
from src.pipeline.optuna_utils import DEFAULT_OPTUNA_DB_URL, create_optuna_study, delete_optuna_study, get_next_study_name
from src.pipeline.slurm import generate_sbatch_header, generate_sbatch_script, submit_sbatch


def run_slurm_training(cfg, args, online_list, offline_list, dataset_list, sanitized_extra_args, storage_url, is_sweep):
    """Submit online and offline training jobs to Slurm cluster.
    
    Returns:
        tuple[list[str], list[str], bool]: (job_ids, eval_commands, is_consolidated)
    """
    log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
    if log_dir.exists():
        print(f"Clearing old logs in {log_dir}...")
        for log_file in log_dir.glob("*"):
            if log_file.is_file():
                try:
                    log_file.unlink()
                except OSError:
                    pass
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path("results/checkpoints") / cfg.group / cfg.experiment_id
    if ckpt_dir.exists():
        print(f"Clearing old checkpoints in {ckpt_dir} for fresh experiment submission...")
        try:
            shutil.rmtree(ckpt_dir)
        except OSError as e:
            print(f"Notice: Could not clear checkpoint dir {ckpt_dir}: {e}")

    should_consolidate = getattr(args, "consolidate", False) or cfg.get("consolidate", False)

    # Consolidated Single 1-GPU Slurm Job Execution
    if should_consolidate:
        print(f"\n=== Preparing Consolidated 1-GPU Slurm Job ({cfg.experiment_id}) ===")
        job_name = f"all_{cfg.experiment_id}"
        cores = args.cores
        script_content = generate_sbatch_header(
            job_name=job_name,
            log_dir=log_dir,
            partition=args.partition,
            gpus=args.gpus,
            cores=cores,
            nodes=args.nodes,
            time=args.time,
            gpu_type=getattr(args, "gpu_type", None),
            gres=getattr(args, "gres", None),
            no_gres=getattr(args, "no_gres", False),
        )
        script_content += f"\nexport PROJECT_ROOT={os.getcwd()}\n"
        script_content += f"export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH\n"
        script_content += f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python\n"
        script_content += f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n\n"

        # 1. Online Training Commands
        if not args.no_online:
            for agent_config in online_list:
                agent_name_internal = normalize_agent_name(agent_config)
                dataset_path = f"in/datasets/{cfg.group}/{cfg.experiment_id}/{agent_name_internal}"
                cmd_args = [
                    "src/train.py",
                    f"+experiment={args.experiment}",
                    f"++local=false",
                    f"mode=online",
                    f"agent={agent_config}",
                    f"++agent.name={agent_name_internal}",
                    f"++dataset_path={dataset_path}"
                ]
                if is_sweep:
                    study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name_internal)
                    direction = cfg.hydra.sweeper.get("direction", "minimize") if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper") else "minimize"
                    create_optuna_study(storage_url, study_name, direction=direction)
                    cmd_args.append(f"++hydra.sweeper.study_name={study_name}")
                cmd_args += sanitized_extra_args
                train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                script_content += f'echo "=== [Phase: Online Training] {agent_config} ==="\n'
                script_content += f"$PROJECT_ROOT/venv/bin/python3 {train_cmd}\n\n"

        # 2. Offline Training Commands (All methods & datasets in parallel)
        if not args.no_offline:
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
                    cmd_args = [
                        "src/train.py",
                        f"+experiment={args.experiment}",
                        f"++local=false",
                        f"mode=offline",
                        f"agent={agent_config}",
                        f"++agent.name={target_agent_name}",
                        f"++mode.dataset_path='{dataset_path}'"
                    ]
                    if cfg.env.name == "pyrenees":
                        ruleset = "default" if dataset_id == "problem" else "step"
                        cmd_args.append(f"++env.rules={ruleset}")
                        cmd_args.append(f"++env.problem_type='{dataset_id}'")
                    if is_sweep:
                        study_name = get_next_study_name(cfg.group, cfg.experiment_id, target_agent_name)
                        direction = cfg.hydra.sweeper.get("direction", "minimize") if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper") else "minimize"
                        create_optuna_study(storage_url, study_name, direction=direction)
                        cmd_args.append(f"++hydra.sweeper.study_name={study_name}")
                        if "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
                            cmd_args.append("--multirun")
                    cmd_args += sanitized_extra_args
                    train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                    if total_runs > 1:
                        script_content += f'echo "  -> Starting [{agent_config}] on [{dataset_id}] in background..."\n'
                        script_content += f"$PROJECT_ROOT/venv/bin/python3 {train_cmd} &\n\n"
                    else:
                        script_content += f"$PROJECT_ROOT/venv/bin/python3 {train_cmd}\n\n"

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
                    script_content += f"$PROJECT_ROOT/venv/bin/python3 -c \"from src.pipeline.optuna_utils import promote_best_trial_checkpoint; promote_best_trial_checkpoint('{cfg.group}', '{cfg.experiment_id}', '{target_agent_name}', '{storage_arg}', '{study_name}')\"\n"
            script_content += '\n'

        # 3. Final Plotting
        if not args.no_plot:
            plot_cmd = f"$PROJECT_ROOT/venv/bin/python3 plot/manager.py {cfg.experiment_id}"
            if args.plot_style:
                plot_cmd += f" --style {args.plot_style}"
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
    if not args.no_online:
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
            overrides_slurm = [
                "src/train.py",
                f"+experiment={args.experiment}",
                f"++local=false",
                f"mode=online",
                f"agent={agent_config}",
                f"++agent.name={agent_name_internal}"
            ]
            if is_sweep:
                study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name_internal)
                direction = cfg.hydra.sweeper.get("direction", "minimize") if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper") else "minimize"
                create_optuna_study(storage_url, study_name, direction=direction)
                overrides_slurm.append(f"++hydra.sweeper.study_name={study_name}")
            else:
                overrides_slurm.append(f"++dataset_path={dataset_path}")
            overrides_slurm += sanitized_extra_args
            
            script_content = generate_sbatch_script(
                job_name, overrides_slurm, log_dir=str(log_dir),
                partition=args.partition, gpus=args.gpus, cores=args.cores, nodes=args.nodes, time=args.time,
                gpu_type=getattr(args, "gpu_type", None), gres=getattr(args, "gres", None),
                no_gres=getattr(args, "no_gres", False),
            )
            job_id = submit_sbatch(script_content)
            if job_id:
                job_ids.append(job_id)
                online_job_ids[agent_config] = job_id
    else:
        print("\n=== Skipping Online Training Phase ===")

    # 2. Offline Training Phases (1 Slurm Job per Method, running datasets in parallel on GPU)
    eval_job_ids = []
    if not args.no_offline:
        for agent_config in offline_list:
            agent_name_internal = normalize_agent_name(agent_config)
            job_name = f"{agent_name_internal}_{cfg.experiment_id}"
            
            # Aggregate dependencies if datasets were generated from online jobs
            deps = [online_job_ids.get(ds) for ds in dataset_list if online_job_ids.get(ds)]
            dependency_str = ":".join(deps) if deps else None

            script_content = generate_sbatch_header(
                job_name=job_name,
                log_dir=log_dir,
                partition=args.partition,
                gpus=args.gpus,
                cores=args.cores,
                nodes=args.nodes,
                time=args.time,
                gpu_type=getattr(args, "gpu_type", None),
                gres=getattr(args, "gres", None),
                no_gres=getattr(args, "no_gres", False),
                dependency=dependency_str,
            )
            script_content += f"\nexport PROJECT_ROOT={os.getcwd()}\n"
            script_content += f"export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH\n"
            script_content += f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python\n"
            script_content += f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n\n"

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
                cmd_args = [
                    "src/train.py",
                    f"+experiment={args.experiment}",
                    f"++local=false",
                    f"mode=offline",
                    f"agent={agent_config}",
                    f"++agent.name={target_agent_name}",
                    f"++mode.dataset_path='{dataset_path}'"
                ]
                if cfg.env.name == "pyrenees":
                    ruleset = "default" if dataset_id == "problem" else "step"
                    cmd_args.append(f"++env.rules={ruleset}")
                    cmd_args.append(f"++env.problem_type='{dataset_id}'")
                if is_sweep:
                    study_name = get_next_study_name(cfg.group, cfg.experiment_id, target_agent_name)
                    direction = cfg.hydra.sweeper.get("direction", "minimize") if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper") else "minimize"
                    create_optuna_study(storage_url, study_name, direction=direction)
                    cmd_args.append(f"++hydra.sweeper.study_name={study_name}")
                    if "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
                        cmd_args.append("--multirun")
                cmd_args += sanitized_extra_args
                
                train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                
                if len(dataset_list) > 1:
                    phase_label = "Offline Sweep (Parallel GPU)" if is_sweep else "Offline Training (Parallel GPU)"
                    script_content += f'echo "=== [Phase: {phase_label}] {agent_config} on {dataset_id} ==="\n'
                    script_content += f"$PROJECT_ROOT/venv/bin/python3 {train_cmd} &\n\n"
                else:
                    phase_label = "Offline Sweep" if is_sweep else "Offline Training"
                    script_content += f'echo "=== [Phase: {phase_label}] {agent_config} on {dataset_id} ==="\n'
                    script_content += f"$PROJECT_ROOT/venv/bin/python3 {train_cmd}\n\n"

            if len(dataset_list) > 1:
                script_content += 'echo "Waiting for all concurrent problem sweeps to complete on GPU..."\nwait\n\n'

            if is_sweep:
                script_content += 'echo "=== Promoting Winning Checkpoints for All Datasets ==="\n'
                storage_arg = storage_url if storage_url else ""
                for d_id in dataset_list:
                    d_name = normalize_agent_name(d_id)
                    t_agent = f"{agent_name_internal}_{d_name}" if len(dataset_list) > 1 else agent_name_internal
                    s_name = f"{cfg.experiment_id}_{t_agent}"
                    script_content += f"$PROJECT_ROOT/venv/bin/python3 -c \"from src.pipeline.optuna_utils import promote_best_trial_checkpoint; promote_best_trial_checkpoint('{cfg.group}', '{cfg.experiment_id}', '{t_agent}', '{storage_arg}', '{s_name}')\"\n"
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
