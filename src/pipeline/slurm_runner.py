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
        script_content += f"export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH\n\n"

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
                    cmd_args.append(f"++hydra.sweeper.study_name={study_name}")
                cmd_args += sanitized_extra_args
                train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                script_content += f'echo "=== [Phase: Online Training] {agent_config} ==="\n'
                script_content += f"$PROJECT_ROOT/venv/bin/python3 {train_cmd}\n\n"

        # 2. Offline Training Commands
        if not args.no_offline:
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

                for agent_config in offline_list:
                    agent_name_internal = normalize_agent_name(agent_config)
                    cmd_args = [
                        "src/train.py",
                        f"+experiment={args.experiment}",
                        f"++local=false",
                        f"mode=offline",
                        f"agent={agent_config}",
                        f"++agent.name={agent_name_internal}",
                        f"++mode.dataset_path={dataset_path}"
                    ]
                    if is_sweep:
                        study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name_internal)
                        cmd_args.append(f"++hydra.sweeper.study_name={study_name}")
                    cmd_args += sanitized_extra_args
                    train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                    script_content += f'echo "=== [Phase: Offline Training (Parallel GPU)] {agent_config} on {dataset_id} ==="\n'
                    script_content += f"$PROJECT_ROOT/venv/bin/python3 {train_cmd} &\n\n"

        script_content += 'echo "Waiting for all concurrent training methods on GPU to complete..."\nwait\n\n'

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
            study_name = get_next_study_name(storage_url, cfg.experiment_id, agent_name_internal)
            
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

    # 2. Offline Training Phases (Many-to-Many)
    eval_job_ids = []
    if not args.no_offline:
        for dataset_id in dataset_list:
            dataset_name_internal = normalize_agent_name(dataset_id)
            is_online = dataset_id in online_list
            dependency_job_id = online_job_ids.get(dataset_id)
            
            for agent_config in offline_list:
                agent_name_internal = normalize_agent_name(agent_config)
                clean_ds = "mimic" if ("mimic" in dataset_name_internal or cfg.group == "mimic") else dataset_name_internal
                if clean_ds in cfg.experiment_id:
                    job_name = f"{agent_name_internal}_{cfg.experiment_id}"
                else:
                    job_name = f"{agent_name_internal}_{clean_ds}_{cfg.experiment_id}"
                overrides_slurm = [
                    "src/train.py",
                    f"+experiment={args.experiment}",
                    f"++local=false",
                    f"mode=offline",
                    f"agent={agent_config}",
                    f"++agent.name={agent_name_internal}"
                ]
                if is_sweep:
                    overrides_slurm.append(f"++hydra.sweeper.study_name={study_name}")
                
                if is_online and is_sweep:
                    storage_url_slurm = storage_url if storage_url else DEFAULT_OPTUNA_DB_URL
                    study_name_slurm = f"{cfg.experiment_id}_{dataset_name_internal}"
                    best_id_cmd = f"BEST_ID=$($PROJECT_ROOT/venv/bin/python3 -c \"import sys; sys.path.append('$PROJECT_ROOT'); from run_pipeline import get_best_trial_id; print(get_best_trial_id('{storage_url_slurm}', '{study_name_slurm}'))\")"
                    dataset_path_cmd = f"D_PATH=in/datasets/{cfg.experiment_id}/{dataset_name_internal}/$BEST_ID"
                    
                    cmd_args = overrides_slurm + sanitized_extra_args
                    train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                    
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
                        dependency=dependency_job_id
                    )
                    script_content += f"\n"
                    script_content += f"export PROJECT_ROOT={os.getcwd()}\n"
                    script_content += f"export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH\n"
                    script_content += f"{best_id_cmd}\n"
                    script_content += f"{dataset_path_cmd}\n"
                    script_content += f"if [ ! -d \"$D_PATH\" ] || [ -z \"$(ls $D_PATH/*.pkl 2>/dev/null)\" ]; then\n"
                    script_content += f"    echo \"ERROR: Best trial dataset not found at $D_PATH for experiment {cfg.experiment_id}.\" >&2\n"
                    script_content += f"    exit 1\n"
                    script_content += f"fi\n"
                    script_content += f"echo \"Using dataset: $D_PATH\"\n"
                    script_content += f"$PROJECT_ROOT/venv/bin/python3 {train_cmd} ++mode.dataset_path=$D_PATH\n"
                else:
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
                    overrides_slurm.append(f"++mode.dataset_path={dataset_path}")
                    overrides_slurm += sanitized_extra_args
                    script_content = generate_sbatch_script(
                        job_name, overrides_slurm, log_dir=str(log_dir),
                        partition=args.partition, gpus=args.gpus, cores=args.cores, nodes=args.nodes,
                        gpu_type=getattr(args, "gpu_type", None), gres=getattr(args, "gres", None),
                        no_gres=getattr(args, "no_gres", False),
                        dependency=dependency_job_id, time=args.time
                    )
                
                job_id = submit_sbatch(script_content)
                if job_id:
                    job_ids.append(job_id)
    else:
        print("\n=== Skipping Offline Training Phase ===")

    job_ids.extend(eval_job_ids)
    return job_ids, eval_commands, False
