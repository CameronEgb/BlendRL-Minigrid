import os
import subprocess
import argparse
import sys
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from pathlib import Path
import re
import datetime

def generate_sbatch_script(job_name, cmd_args, log_dir, partition="rtx4060ti16g", gpus=1, cores=16, nodes=1, dependency=None):
    script = f"#!/bin/bash\n"
    script += f"#SBATCH --job-name={job_name}\n"
    script += f"#SBATCH --partition={partition}\n"
    script += f"#SBATCH --ntasks-per-node={cores}\n"
    script += f"#SBATCH --nodes={nodes}\n"
    script += f"#SBATCH --output={log_dir}/%x_%j.out\n"
    script += f"#SBATCH --error={log_dir}/%x_%j.err\n"
    
    if dependency:
        script += f"#SBATCH --dependency=afterok:{dependency}\n"
        
    script += f"\n"
    script += f"export PROJECT_ROOT={os.getcwd()}\n"
    script += f"export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH\n"
    
    # Construct the python command with absolute venv path
    cmd_str = "$PROJECT_ROOT/venv/bin/python3 " + " ".join(cmd_args)
    script += f"echo 'Running: {cmd_str}'\n"
    script += f"{cmd_str}\n"
    
    return script

def submit_sbatch(script_content):
    print(f"Submitting Slurm job via stdin...")
    try:
        # Pipe script_content directly to sbatch stdin
        result = subprocess.run(
            ["sbatch"], 
            input=script_content, 
            capture_output=True, 
            text=True, 
            check=True
        )
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            print(f"-> Job ID: {job_id}")
            return job_id
        else:
            print(f"Could not parse job ID from: {result.stdout}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: 'sbatch' command not found. Are you on the Slurm cluster?")
        return "99999"

def main():
    parser = argparse.ArgumentParser(description="NeSyRL Slurm Pipeline")
    parser.add_argument("experiment", type=str, help="Experiment name from conf/experiment/")
    parser.add_argument("--partition", type=str, default="rtx4060ti16g", help="Slurm partition")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per job")
    parser.add_argument("--cores", type=int, default=16, help="Number of CPU cores per job")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes per job")
    parser.add_argument("--plot-style", type=str, default=None, help="Style config for plotter")
    args, extra_args = parser.parse_known_args()
    
    sanitized_extra_args = []
    for arg in extra_args:
        if "=" in arg and not (arg.startswith("+") or arg.startswith("++")):
            sanitized_extra_args.append("++" + arg)
        else:
            sanitized_extra_args.append(arg)
            
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="conf")
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"] + sanitized_extra_args)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
        
    online_methods = cfg.get("online_methods", "")
    offline_methods = cfg.get("offline_methods", "")
    offline_datasets = cfg.get("offline_datasets", "")
    
    def parse_list(val):
        if not val: return []
        if isinstance(val, (list, tuple)): return list(val)
        return [item.strip() for item in str(val).split(",") if item.strip()]

    online_list = parse_list(online_methods)
    offline_list = parse_list(offline_methods)
    dataset_list = parse_list(offline_datasets)
    
    print(f"Detected Online Methods: {online_list}")
    print(f"Detected Offline Methods: {offline_list}")
    
    if not dataset_list:
        dataset_list = online_list if online_list else ["ppo"]
    
    print(f"Using Datasets for Offline Training: {dataset_list}")
        
    # Ensure slurm log directories exist
    log_dir = Path("results/logs/slurm") / args.experiment
    if log_dir.exists():
        print(f"Clearing old logs in {log_dir}...")
        for log_file in log_dir.glob("*"):
            if log_file.is_file():
                log_file.unlink()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    job_ids = []
    online_job_ids = {} # dataset_id -> job_id
    
    # 1. Online Training Phases
    for agent_config in online_list:
        print(f"\n=== Preparing Slurm Job: Online Training ({agent_config}) ===")
        agent_name_internal = agent_config.replace("/", "_")
        job_name = f"on_{agent_name_internal}_{args.experiment}"
        
        overrides = [
            "train.py",
            f"+experiment={args.experiment}",
            f"++local=false",
            f"mode=online",
            f"agent={agent_config}",
            f"++agent.name={agent_name_internal}",
            f"++dataset_path=results/datasets/{args.experiment}/{agent_name_internal}"
        ] + sanitized_extra_args
        
        script_content = generate_sbatch_script(
            job_name, overrides, log_dir=str(log_dir),
            partition=args.partition, gpus=args.gpus, cores=args.cores, nodes=args.nodes
        )
        job_id = submit_sbatch(script_content)
        
        if job_id:
            job_ids.append(job_id)
            online_job_ids[agent_config] = job_id
            
    # 2. Offline Training Phases
    for dataset_id in dataset_list:
        dataset_name_internal = dataset_id.replace("/", "_")
        is_online = dataset_id in online_list
        dataset_path = Path("results/datasets") / args.experiment / dataset_name_internal
        
        # Dependency logic
        dependency_job_id = online_job_ids.get(dataset_id)
        
        if not is_online and not dataset_path.exists():
            print(f"Error: Dataset '{dataset_id}' not found.")
            sys.exit(1)
            
        for agent_config in offline_list:
            print(f"\n=== Preparing Slurm Job: Offline Training ({agent_config}) on Dataset ({dataset_id}) ===")
            agent_name_internal = agent_config.replace("/", "_")
            job_name = f"off_{agent_name_internal}_{dataset_name_internal}_{args.experiment}"
            
            dataset_path_override = any("mode.dataset_path=" in arg for arg in sanitized_extra_args)
            overrides = [
                "train.py",
                f"+experiment={args.experiment}",
                f"++local=false",
                f"mode=offline",
                f"agent={agent_config}",
                f"++agent.name={agent_name_internal}"
            ]
            if not dataset_path_override:
                overrides.append(f"++mode.dataset_path=results/datasets/{args.experiment}/{dataset_name_internal}")
            overrides += sanitized_extra_args
            
            script_content = generate_sbatch_script(
                job_name, overrides, log_dir=str(log_dir),
                partition=args.partition, gpus=args.gpus, cores=args.cores, nodes=args.nodes,
                dependency=dependency_job_id
            )
            job_id = submit_sbatch(script_content)
            
            if job_id:
                job_ids.append(job_id)

    # 3. Final Dependent Job: Plotting and Syncing
    if job_ids:
        print(f"\n=== Preparing Final Job: Plotting and Syncing ({args.experiment}) ===")
        job_name = f"final_{args.experiment}"
        
        all_dependencies = ":".join(jid for jid in job_ids if jid != "99999")
        
        project_root = os.getcwd()
        plot_cmd = f"{project_root}/venv/bin/python3 plot_results.py {args.experiment}"
        if args.plot_style:
            plot_cmd += f" --style {args.plot_style}"
            
        sync_cmd = f"rsync -avz results/plots/{args.experiment}/ ${{MAC_USER}}@${{MAC_IP}}:${{MAC_PATH}}"
        
        final_script = f"#!/bin/bash\n"
        final_script += f"#SBATCH --job-name={job_name}\n"
        final_script += f"#SBATCH --partition={args.partition}\n"
        final_script += f"#SBATCH --ntasks-per-node=1\n"
        final_script += f"#SBATCH --nodes=1\n"
        final_script += f"#SBATCH --output={log_dir}/%x_%j.out\n"
        final_script += f"#SBATCH --error={log_dir}/%x_%j.err\n"
        if all_dependencies:
            final_script += f"#SBATCH --dependency=afterany:{all_dependencies}\n"
        final_script += f"\nexport PROJECT_ROOT={project_root}\n"
        final_script += f"export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH\n\n"
        final_script += f"echo 'Generating final plots...'\n"
        final_script += f"{plot_cmd}\n"
        final_script += f"echo 'Attempting to sync plots to Mac...'\n"
        final_script += f"if [ -z \"$MAC_USER\" ] || [ -z \"$MAC_IP\" ] || [ -z \"$MAC_PATH\" ]; then\n"
        final_script += f"  echo 'Error: MAC_USER, MAC_IP, or MAC_PATH not set. Skipping sync.'\n"
        final_script += f"else\n"
        final_script += f"  {sync_cmd}\n"
        final_script += f"fi\n"
        
        submit_sbatch(final_script)
                
    print(f"\n=== Submission Complete ===")

if __name__ == "__main__":
    main()
