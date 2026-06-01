import os
import subprocess
import argparse
import sys
import webbrowser
import time
import threading
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from pathlib import Path

def get_best_trial_id(storage_url, study_name):
    """Queries the optuna database to find the best trial ID for a given study."""
    if not storage_url or not storage_url.startswith("sqlite:///"):
        return "0" # Default to 0 if not using sqlite
    
    db_path = storage_url.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        return "0"

    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get study_id first
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return "0"
        
        study_id = result[0]
        
        # Find trial with max value for this study
        # In Optuna 3.x/4.x values are in trial_values table
        cursor.execute("""
            SELECT t.number 
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE' 
            ORDER BY tv.value DESC LIMIT 1
        """, (study_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result is not None:
            return str(result[0])
    except Exception as e:
        print(f"Warning: Could not query best trial from Optuna: {e}")
    
    return "0"

def get_python_executable():
    """Returns the path to the python executable to use for all subprocesses."""
    # If we are already running in the venv, use that
    # Otherwise, look for venv/bin/python3
    project_root = os.getcwd()
    venv_python = os.path.join(project_root, "venv", "bin", "python3")
    
    # Check if we should use python3.13 specifically if it exists (since all dependencies are there)
    venv_python_13 = os.path.join(project_root, "venv", "bin", "python3.13")
    if os.path.exists(venv_python_13):
        return venv_python_13
        
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable

def launch_optuna_dashboard(storage_url):
    """Launches the optuna-dashboard in the background and opens the browser."""
    if not storage_url:
        return

    # Ensure directory exists if using sqlite
    if storage_url.startswith("sqlite:///"):
        db_path = storage_url.replace("sqlite:///", "")
        if "/" in db_path:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Check if a dashboard is already running for this storage_url
    try:
        ps_output = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        if storage_url in ps_output and "optuna_dashboard" in ps_output:
            print(f"Optuna Dashboard already running for: {storage_url}. Skipping launch.")
            return
    except:
        pass

    print(f"Launching Optuna Dashboard for: {storage_url}")

    # Use managed python to ensure dashboard is available
    venv_python = get_python_executable()
    venv_bin_dir = os.path.dirname(venv_python)
    optuna_dashboard_bin = os.path.join(venv_bin_dir, "optuna-dashboard")

    # Try to launch optuna-dashboard
    try:
        # Launch in background
        port = 8080
        # Try different ports if 8080 is taken
        for p in range(8080, 8090):
            try:
                # Use a dummy check to see if port is in use
                subprocess.run(["nc", "-z", "localhost", str(p)], capture_output=True, check=True)
                continue # Port is in use
            except:
                port = p
                break

        # Ensure logging directory exists
        log_file_path = "results/optuna/dashboard.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        log_file = open(log_file_path, "w")

        cmd = [optuna_dashboard_bin, storage_url, "--port", str(port)]
        subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        print(f"Optuna Dashboard started at http://127.0.0.1:{port}")

        # Wait a moment for server to start then open browser in a separate thread
        def open_browser():
            time.sleep(2)
            url = f"http://127.0.0.1:{port}"
            if sys.platform == "darwin":
                # On macOS, -g opens in background without focusing
                subprocess.run(["open", "-g", url])
            else:
                webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: optuna-dashboard could not be started: {e}")
        print("Please ensure 'optuna-dashboard' is installed in your environment.")


def delete_optuna_study(storage_url, study_name):
    """Deletes an existing study from the Optuna database to start fresh."""
    if not storage_url or not storage_url.startswith("sqlite:///"):
        return
        
    venv_python = get_python_executable()
    cmd = [
        venv_python, "-c",
        f"import optuna; "
        f"optuna.delete_study(study_name='{study_name}', storage='{storage_url}')"
    ]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print(f"Overwriting existing Optuna study: {study_name}")
    except subprocess.CalledProcessError:
        pass # Study didn't exist or other error

def run_experiment(overrides):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    venv_python = get_python_executable()
    cmd = [venv_python, "train.py"] + overrides
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def run_plotting(experiment, style=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
    
    venv_python = get_python_executable()
    cmd = [venv_python, "plot_results.py", experiment]
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
    parser.add_argument("--no-dash", action="store_true", help="Disable the automatic dashboard")
    parser.add_argument("--dash-only", action="store_true", help="Launch the persistent dashboard and exit")
    args, extra_args = parser.parse_known_args()

    # Prepare extra_args for subprocesses and compose call by ensuring they use ++ for robust overrides
    sanitized_extra_args = []
    overrides_for_compose = [f"+experiment={args.experiment}"]
    
    for arg in extra_args:
        if "=" in arg:
            if not (arg.startswith("+") or arg.startswith("++")):
                sanitized_arg = "++" + arg
            else:
                sanitized_arg = arg
            sanitized_extra_args.append(sanitized_arg)
            overrides_for_compose.append(sanitized_arg)
        else:
            # Flags like --multirun or -m should be passed to subprocess but NOT to compose
            sanitized_extra_args.append(arg)

    # Load configuration to get method lists and Optuna info
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="conf")
        # return_hydra_config=True is REQUIRED to see the hydra.sweeper node
        cfg = compose(config_name="config", overrides=overrides_for_compose, return_hydra_config=True)
        
        # Check for Optuna storage in the config
        storage_url = None
        is_sweep = "--multirun" in sanitized_extra_args or "-m" in sanitized_extra_args
        
        if "hydra" in cfg and "sweeper" in cfg.hydra and "storage" in cfg.hydra.sweeper:
            storage_url = cfg.hydra.sweeper.storage
            
        if storage_url and (is_sweep or not args.no_dash):
            # Resolve interpolation if any
            storage_url = storage_url.replace("${experiment_id}", args.experiment)
            launch_optuna_dashboard(storage_url)
            
            if args.dash_only:
                print("Dashboard running in persistent mode. Press Ctrl+C to exit.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    sys.exit(0)
                    
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
        
    if args.dash_only:
        print("Error: Could not find Optuna storage URL in configuration.")
        sys.exit(1)

    online_methods = cfg.get("online_methods", "")
    offline_methods = cfg.get("offline_methods", "")
    offline_datasets = cfg.get("offline_datasets", "")

    # Helper to parse comma-separated lists or OmegaConf lists
    def parse_list(val):
        if not val: return []
        if isinstance(val, (list, tuple)): return list(val)
        # Handle OmegaConf ListConfig/DictConfig
        if hasattr(val, "__iter__") and not isinstance(val, str):
            return list(val)
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

    # Track the best trial ID for each online method to use its dataset later
    best_online_trial_ids = {}

    # 1. Online Training Phases
    for agent_config in online_list:
        print(f"\n=== Phase: Online Training ({agent_config}) ===")
        # Standardize agent name by replacing / with _ for the internal agent.name field
        # this ensures loggers create manageable folder structures or names.
        agent_name_internal = agent_config.replace("/", "_")
        study_name = f"{args.experiment}_{agent_name_internal}"
        
        overrides = [
            f"+experiment={args.experiment}",
            f"++local={str(args.local).lower()}",
            f"mode=online",
            f"agent={agent_config}",
            f"++agent.name={agent_name_internal}",
            f"++dataset_path=results/datasets/{cfg.group}/{args.experiment}/{agent_name_internal}",
            f"++hydra.sweeper.study_name={study_name}"
        ] + sanitized_extra_args
        
        if is_sweep:
            delete_optuna_study(storage_url, study_name)
            
        run_experiment(overrides)
        
        # After training, find the best trial ID if we were sweeping
        if is_sweep:
            best_id = get_best_trial_id(storage_url, study_name)
            best_online_trial_ids[agent_config] = best_id
            print(f"Best trial for {agent_config} was ID: {best_id}")
        else:
            best_online_trial_ids[agent_config] = "0"

    # 2. Offline Training Phases (Many-to-Many)
    for dataset_id in dataset_list:
        # Standardize dataset name as well
        dataset_name_internal = dataset_id.replace("/", "_")
        
        # Determine the correct dataset path, favoring the best trial if found
        best_id = best_online_trial_ids.get(dataset_id, "0")
        dataset_path = Path("results/datasets") / cfg.group / args.experiment / dataset_name_internal / best_id
        
        # Fallback for datasets that might exist but weren't part of this run's online methods
        if not dataset_path.exists():
            # Check if there's a non-trial version (legacy or non-sweep)
            alt_path = Path("results/datasets") / cfg.group / args.experiment / dataset_name_internal
            if alt_path.exists() and any(alt_path.glob("*.pkl")):
                 dataset_path = alt_path
            else:
                print(f"Error: Dataset '{dataset_id}' not found at {dataset_path} or {alt_path}")
                sys.exit(1)

        print(f"Using dataset from: {dataset_path}")

        for agent_config in offline_list:
            agent_name_internal = agent_config.replace("/", "_")
            print(f"\n=== Phase: Offline Training ({agent_config}) on Dataset ({dataset_id}) ===")
            
            # Check if a custom dataset path is already in extra_args or config
            dataset_path_override = any("mode.dataset_path=" in arg for arg in sanitized_extra_args)
            
            study_name = f"{args.experiment}_{agent_name_internal}_{dataset_name_internal}"
            overrides = [
                f"+experiment={args.experiment}",
                f"++local={str(args.local).lower()}",
                f"mode=offline",
                f"agent={agent_config}",
                f"++agent.name={agent_name_internal}",
                f"++hydra.sweeper.study_name={study_name}"
            ]
            
            if not dataset_path_override:
                overrides.append(f"++mode.dataset_path={dataset_path}")
                
            overrides += sanitized_extra_args
            
            if is_sweep:
                delete_optuna_study(storage_url, study_name)
                
            run_experiment(overrides)

    # 4. Plotting
    if not args.no_plot:
        run_plotting(args.experiment, style=args.plot_style)

if __name__ == "__main__":
    main()
