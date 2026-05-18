import os
import subprocess
import argparse
import sys
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from pathlib import Path

def run_experiment(overrides):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        
    cmd = [sys.executable, "train.py"] + overrides
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def run_plotting(experiment, style=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
    
    cmd = [sys.executable, "plot_results.py", experiment]
    if style:
        cmd += ["--style", style]
        
    print(f"\n=== Generating Plots for experiment: {experiment} ===")
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser(description="NeSyRL Experiment Pipeline")
    parser.add_argument("experiment", type=str, help="Experiment name from conf/experiment/")
    parser.add_argument("--local", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_true", help="Skip automatic plotting")
    parser.add_argument("--plot-style", type=str, default=None, help="Style config for plotter")
    args, extra_args = parser.parse_known_args()

    # Prepare extra_args for subprocesses and compose call by ensuring they use ++ for robust overrides
    sanitized_extra_args = []
    for arg in extra_args:
        if "=" in arg and not (arg.startswith("+") or arg.startswith("++")):
            sanitized_extra_args.append("++" + arg)
        else:
            sanitized_extra_args.append(arg)

    # Load configuration to get method lists
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="conf")
        # Include extra_args in compose to allow overriding online_methods/offline_methods from CLI
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"] + sanitized_extra_args)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    online_methods = cfg.get("online_methods", "")
    offline_methods = cfg.get("offline_methods", "")
    offline_datasets = cfg.get("offline_datasets", "")

    # Helper to parse comma-separated lists
    def parse_list(val):
        if not val: return []
        if isinstance(val, (list, tuple)): return list(val)
        return [item.strip() for item in str(val).split(",") if item.strip()]

    online_list = parse_list(online_methods)
    offline_list = parse_list(offline_methods)
    dataset_list = parse_list(offline_datasets)

    print(f"Detected Online Methods: {online_list}")
    print(f"Detected Offline Methods: {offline_list}")
    
    # If offline_datasets is not specified, default to using all online_methods as datasets
    if not dataset_list:
        dataset_list = online_list if online_list else ["ppo"]
    
    print(f"Using Datasets for Offline Training: {dataset_list}")

    # 1. Online Training Phases
    for agent_config in online_list:
        print(f"\n=== Phase: Online Training ({agent_config}) ===")
        # Standardize agent name by replacing / with _ for the internal agent.name field
        # this ensures loggers create manageable folder structures or names.
        agent_name_internal = agent_config.replace("/", "_")
        
        overrides = [
            f"+experiment={args.experiment}",
            f"++local={str(args.local).lower()}",
            f"mode=online",
            f"agent={agent_config}",
            f"++agent.name={agent_name_internal}",
            f"++dataset_path=results/datasets/{cfg.group}/{args.experiment}/{agent_name_internal}"
        ] + sanitized_extra_args
        run_experiment(overrides)

    # 2. Offline Training Phases (Many-to-Many)
    for dataset_id in dataset_list:
        # Standardize dataset name as well
        dataset_name_internal = dataset_id.replace("/", "_")
        
        # Validate dataset existence or planned creation
        is_online = dataset_id in online_list
        dataset_path = Path("results/datasets") / cfg.group / args.experiment / dataset_name_internal
        
        if not is_online and not dataset_path.exists():
            print(f"Error: Dataset '{dataset_id}' not found.")
            print(f"It is not in the current online_methods list and no folder exists at {dataset_path}")
            sys.exit(1)

        for agent_config in offline_list:
            agent_name_internal = agent_config.replace("/", "_")
            print(f"\n=== Phase: Offline Training ({agent_config}) on Dataset ({dataset_id}) ===")
            
            # Check if a custom dataset path is already in extra_args or config
            dataset_path_override = any("mode.dataset_path=" in arg for arg in sanitized_extra_args)
            
            overrides = [
                f"+experiment={args.experiment}",
                f"++local={str(args.local).lower()}",
                f"mode=offline",
                f"agent={agent_config}",
                f"++agent.name={agent_name_internal}"
            ]
            
            if not dataset_path_override:
                overrides.append(f"++mode.dataset_path=results/datasets/{cfg.group}/{args.experiment}/{dataset_name_internal}")
                
            overrides += sanitized_extra_args
            run_experiment(overrides)

    # 4. Plotting
    if not args.no_plot:
        run_plotting(args.experiment, style=args.plot_style)

if __name__ == "__main__":
    main()
