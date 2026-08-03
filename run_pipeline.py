import os
import subprocess
import argparse
import sys
import webbrowser
import time
import threading
import re
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from pathlib import Path
from plot.base import clean_label

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
        # We try a few common ways Optuna stores this to be robust
        try:
            cursor.execute("""
                SELECT t.number 
                FROM trials t
                JOIN trial_values tv ON t.trial_id = tv.trial_id
                WHERE t.study_id = ? AND t.state = 'COMPLETE' 
                ORDER BY tv.value DESC LIMIT 1
            """, (study_id,))
            result = cursor.fetchone()
        except:
            # Fallback for different Optuna schema versions
            cursor.execute("""
                SELECT t.number 
                FROM trials t
                WHERE t.study_id = ? AND t.state = 'COMPLETE' 
                ORDER BY t.trial_id DESC LIMIT 1
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
    
    # Check if we should use python3.13 specifically if it exists
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

def create_optuna_study(storage_url, study_name):
    """Pre-creates/initializes an Optuna study to avoid schema initialization race conditions on cluster nodes."""
    if not storage_url or not storage_url.startswith("sqlite:///"):
        return
        
    venv_python = get_python_executable()
    cmd = [
        venv_python, "-c",
        f"import optuna; "
        f"optuna.create_study(study_name='{study_name}', storage='{storage_url}', load_if_exists=True)"
    ]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print(f"Pre-initialized Optuna study: {study_name}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to pre-initialize Optuna study '{study_name}': {e.stderr.decode().strip()}")

def run_experiment(overrides):
    env = os.environ.copy()
    project_root = os.getcwd()
    new_paths = [
        os.path.abspath("src"),
        os.path.join(project_root, "src", "fyd_repo", "src")
    ]
    env["PYTHONPATH"] = ":".join(new_paths) + ":" + env.get("PYTHONPATH", "")
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    venv_python = get_python_executable()
    cmd = [venv_python, "src/train.py"] + overrides
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def run_plotting(experiment, style=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".") + ":" + os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
    
    venv_python = get_python_executable()
    cmd = [venv_python, "plot/manager.py", experiment]
        
    print(f"\n=== Auto-Generating Modular Plots for experiment: {experiment} ===")
    subprocess.run(cmd, check=True, env=env)

def find_dataset_globally(agent_name_internal):
    datasets_root = Path("in/datasets")
    if not datasets_root.exists():
        return None
        
    matches = []
    for root, dirs, files in os.walk(datasets_root):
        if any(f.endswith(".pkl") for f in files):
            parts = Path(root).parts
            if agent_name_internal in parts:
                matches.append(root)
                
    if not matches:
        return None
        
    # Sort matches by path depth (fewer parts first) to prefer shallowest path
    matches.sort(key=lambda p: len(Path(p).parts))
    return matches[0]

def generate_sbatch_script(job_name, cmd_args, log_dir, partition="rtx4060ti16g", gpus=1, cores=16, nodes=1, dependency=None, time="04:00:00"):
    script = f"#!/bin/bash\n"
    script += f"#SBATCH --job-name={job_name}\n"
    script += f"#SBATCH --partition={partition}\n"
    script += f"#SBATCH --time={time}\n"
    script += f"#SBATCH --ntasks-per-node={cores}\n"
    script += f"#SBATCH --nodes={nodes}\n"
    script += f"#SBATCH --output={log_dir}/%x_%j.out\n"
    script += f"#SBATCH --error={log_dir}/%x_%j.err\n"
    script += f"#SBATCH --mail-type=END,FAIL\n"
    script += f"#SBATCH --mail-user=cegbert@ncsu.edu\n"

    if dependency:
        script += f"#SBATCH --dependency=afterok:{dependency}\n"
        
    script += f"\n"
    script += f"export PROJECT_ROOT={os.getcwd()}\n"
    script += f"export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/nsfr:$PROJECT_ROOT/src/nudge:$PROJECT_ROOT/src/neumann:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH\n"
    
    # Construct the python command with absolute venv path
    import shlex
    cmd_str = "$PROJECT_ROOT/venv/bin/python3 " + " ".join(shlex.quote(arg) for arg in cmd_args)
    script += f"echo \"Running: {cmd_str}\"\n"
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

def run_early_prediction_eval(checkpoint_path, remake=False):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
    
    venv_python = get_python_executable()
    cmd = [
        venv_python, "src/early_prediction/eval.py",
        "--checkpoint", str(checkpoint_path)
    ]
    if remake:
        cmd.append("--remake")
    print(f"\n=== Running Early Prediction Evaluation for checkpoint: {checkpoint_path} ===")
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser(description="NeSyRL Unified Experiment Pipeline")
    parser.add_argument("experiment", type=str, help="Experiment name from in/config/experiment/")
    parser.add_argument("--local", type=str, default=None, help="Force local run (true/false)")
    parser.add_argument("--partition", type=str, default="rtx4060ti16g", help="Slurm partition")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per job")
    parser.add_argument("--cores", type=int, default=16, help="Number of CPU cores per job")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes per job")
    parser.add_argument("--plot-style", type=str, default=None, help="Style config for plotter")
    parser.add_argument("--no-plot", action="store_true", help="Skip automatic plotting")
    parser.add_argument("--no-online", action="store_true", help="Skip online training phase")
    parser.add_argument("--no-offline", action="store_true", help="Skip offline training phase")
    parser.add_argument("--dash", action="store_true", help="Launch the Optuna dashboard during the run")
    parser.add_argument("--dash-only", action="store_true", help="Launch the persistent dashboard and exit")
    parser.add_argument("--remake", action="store_true", help="Force recalculation and overwrite of early prediction summaries")
    args, extra_args = parser.parse_known_args()

    # Prepare extra_args:
    # sanitized_extra_args -> passed to train.py
    # overrides_for_compose -> passed to hydra.compose to read the config
    sanitized_extra_args = []
    overrides_for_compose = [f"+experiment={args.experiment}"]
    
    for arg in extra_args:
        # Skip local overrides as it is explicitly handled by the parser/sbatch generator
        if "local=" in arg:
            continue
            
        if "=" in arg:
            # If it has a slash, it is a config group (e.g. hydra/sweeper=optuna), do not prepend ++
            # Also, do not prepend ++ to sweep parameters (containing interval, choice, or range)
            is_sweep_param = any(sw in arg for sw in ["interval(", "choice(", "range("])
            if not (arg.startswith("+") or arg.startswith("++") or "/" in arg.split("=")[0] or is_sweep_param or arg.startswith("hydra.")):
                sanitized_arg = "++" + arg
            else:
                sanitized_arg = arg
            sanitized_extra_args.append(sanitized_arg)
            # Exclude sweep parameters and hydra internal configs from compose configuration overrides
            is_sweep = any(sw in sanitized_arg for sw in ["interval(", "choice(", "range("])
            is_hydra = "hydra/" in sanitized_arg or "hydra." in sanitized_arg
            if not (is_sweep or is_hydra):
                overrides_for_compose.append(sanitized_arg)
        else:
            # Flags like --multirun or -m should be passed to subprocess but NOT to compose
            sanitized_extra_args.append(arg)

    # Load configuration to get method lists and Optuna info
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="in/config")
        # return_hydra_config=True is REQUIRED to see the hydra.sweeper node
        cfg = compose(config_name="config", overrides=overrides_for_compose, return_hydra_config=True)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Determine execution mode (local vs slurm cluster)
    local_val = cfg.get("local", True)
    if args.local is not None:
        local_val = args.local.lower() in ("true", "1", "yes")

    print(f"Execution Mode: {'Local' if local_val else 'Slurm Cluster'}")

    # Check for Optuna storage in the config
    storage_url = None
    is_sweep = "--multirun" in sanitized_extra_args or "-m" in sanitized_extra_args
    
    if "hydra" in cfg and "sweeper" in cfg.hydra and "storage" in cfg.hydra.sweeper:
        storage_url = cfg.hydra.sweeper.storage
        os.makedirs("results/optuna", exist_ok=True)
        
    if local_val and storage_url and (args.dash or args.dash_only):
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
                
    if args.dash_only:
        print("Error: Could not find Optuna storage URL in configuration.")
        sys.exit(1)

    online_methods = cfg.get("online_methods", "")
    offline_methods = cfg.get("offline_methods", "")
    offline_datasets = cfg.get("offline_datasets", "")

    # Helper to parse lists
    def parse_list(val):
        if not val: return []
        if isinstance(val, (list, tuple)): return list(val)
        if hasattr(val, "__iter__") and not isinstance(val, str):
            return list(val)
        return [item.strip() for item in str(val).split(",") if item.strip()]

    online_list = parse_list(online_methods)
    offline_list = parse_list(offline_methods)
    dataset_list = parse_list(offline_datasets)

    print(f"Detected Online Methods: {online_list}")
    print(f"Detected Offline Methods: {offline_list}")
    
    # Auto-convert MIMIC dataset if needed
    if cfg.env.name == "mimic":
        cql_dir = Path("in/datasets/mimic/cql")
        if not (cql_dir.exists() and any(cql_dir.glob("*.pkl"))):
            print("\n=== Auto-Converting MIMIC NPZ Dataset (mimic_lazy_0_interventions_balanced.npz) to PKL Format ===")
            subprocess.run([sys.executable, "scripts/convert_npz_to_pkl.py"], check=True)

    print(f"Using Datasets for Offline Training: {dataset_list}")

    # Slurm log setup
    job_ids = []
    online_job_ids = {} # dataset_id -> job_id
    log_dir = None
    if not local_val:
        log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
        if log_dir.exists():
            print(f"Clearing old logs in {log_dir}...")
            for log_file in log_dir.glob("*"):
                if log_file.is_file():
                    log_file.unlink()
        log_dir.mkdir(parents=True, exist_ok=True)

    # Check for early_prediction task types (standalone sweeps or evaluations)
    task_name = cfg.get("task", "")
    if task_name.startswith("early_prediction"):
        def build_early_pred_args(c):
            ep_c = c.get("early_prediction", {})
            a_list = []
            flag_map = {
                "checkpoint": "--checkpoint",
                "dataset_path": "--dataset-path",
                "output_dir": "--output-dir",
                "tune_dir": "--tune-dir",
                "use_tuned_params": "--use-tuned-params",
                "save_checkpoints": "--save-checkpoints",
                "n_splits": "--n-splits",
                "tau_min": "--tau-min",
                "tau_max": "--tau-max",
                "tau_step": "--tau-step",
                "window_hours": "--window-hours",
                "epochs": "--epochs",
                "batch_size": "--batch-size",
                "lr": "--lr",
                "d_model": "--d-model",
                "nhead": "--nhead",
                "n_layers": "--n-layers",
            }
            for k, flag in flag_map.items():
                if k in ep_c and ep_c[k] is not None:
                    if isinstance(ep_c[k], bool):
                        if ep_c[k]:
                            a_list.append(flag)
                    else:
                        a_list.extend([flag, str(ep_c[k])])
            if ep_c.get("use_all_history", False):
                a_list.append("--use-all-history")
            if ep_c.get("use_all_trajectories", False):
                a_list.append("--use-all-trajectories")
            if ep_c.get("no_norm", False):
                a_list.append("--no-norm")
            return a_list

        if task_name in ("early_prediction_sweep", "early_prediction"):
            print(f"\n=== Running Early Prediction Deep Learning Sweep ({cfg.experiment_id}) ===")
            ep_args = build_early_pred_args(cfg)
            if local_val:
                python_exe = get_python_executable()
                cmd = [python_exe, "-u", "src/early_prediction/model.py", "--exp-id", cfg.experiment_id] + ep_args
                print(f"Executing: {' '.join(cmd)}")
                res = subprocess.run(cmd)
                sys.exit(res.returncode)
            else:
                slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
                slurm_dir.mkdir(parents=True, exist_ok=True)
                slurm_script_path = slurm_dir / "early_pred_sweep.slurm"
                cmd_str = " ".join([f'"{arg}"' if " " in arg else arg for arg in ep_args])
                script_content = f"""#!/bin/bash
#SBATCH --job-name=early_pred_{cfg.experiment_id}
#SBATCH --partition=rtx4060ti8g
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --output=results/logs/slurm/{cfg.group}/{cfg.experiment_id}/early_pred_%j.out
#SBATCH --error=results/logs/slurm/{cfg.group}/{cfg.experiment_id}/early_pred_%j.err

echo "=== Sepsis Early Prediction Sweep Execution Start ==="
echo "Node: $(hostname)"
date

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/nsfr:$PROJECT_ROOT/src/nudge:$PROJECT_ROOT/src/neumann:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

mkdir -p results/plots/early_prediction
mkdir -p results/logs

$PROJECT_ROOT/venv/bin/python3 -u src/early_prediction/model.py \\
    --exp-id "{cfg.experiment_id}" \\
    {cmd_str}

echo "=== Sepsis Early Prediction Sweep Execution End ==="
date
"""
                with open(slurm_script_path, "w") as f:
                    f.write(script_content)
                print(f"Submitting Early Prediction Sweep SLURM Job: {slurm_script_path}")
                res = subprocess.run(["sbatch", str(slurm_script_path)], capture_output=True, text=True)
                print(res.stdout)
                if res.stderr:
                    print(res.stderr)
                sys.exit(res.returncode)

        elif task_name == "early_prediction_eval":
            print(f"\n=== Running Early Prediction Checkpoint Evaluation ({cfg.experiment_id}) ===")
            ep_cfg = cfg.get("early_prediction", {})
            ckpt = ep_cfg.get("checkpoint", f"results/checkpoints/{cfg.group}/{cfg.experiment_id}")
            if local_val:
                python_exe = get_python_executable()
                cmd = [python_exe, "-u", "src/early_prediction/eval.py", "--checkpoint", str(ckpt), "--remake"]
                print(f"Executing: {' '.join(cmd)}")
                res = subprocess.run(cmd)
                sys.exit(res.returncode)
            else:
                slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
                slurm_dir.mkdir(parents=True, exist_ok=True)
                slurm_script_path = slurm_dir / "early_pred_eval.slurm"
                script_content = f"""#!/bin/bash
#SBATCH --job-name=eval_pred_{cfg.experiment_id}
#SBATCH --partition=rtx4060ti8g
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --output=results/logs/slurm/{cfg.group}/{cfg.experiment_id}/eval_pred_%j.out
#SBATCH --error=results/logs/slurm/{cfg.group}/{cfg.experiment_id}/eval_pred_%j.err

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/nsfr:$PROJECT_ROOT/src/nudge:$PROJECT_ROOT/src/neumann:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

$PROJECT_ROOT/venv/bin/python3 -u src/early_prediction/eval.py --checkpoint "{ckpt}" --remake
"""
                with open(slurm_script_path, "w") as f:
                    f.write(script_content)
                print(f"Submitting Early Prediction Eval SLURM Job: {slurm_script_path}")
                res = subprocess.run(["sbatch", str(slurm_script_path)], capture_output=True, text=True)
                print(res.stdout)
                if res.stderr:
                    print(res.stderr)
                sys.exit(res.returncode)

        elif task_name in ("early_prediction_tune", "early_prediction_optuna"):
            print(f"\n=== Running Early Prediction Modular Optuna Hyperparameter Search ({cfg.experiment_id}) ===")
            ep_cfg = cfg.get("early_prediction", {})
            n_trials = ep_cfg.get("n_trials", 50)
            ckpt = ep_cfg.get("checkpoint", "results/checkpoints/mimic/tune_mimic_all")
            dataset_path = ep_cfg.get("dataset_path", "in/datasets/mimic/mimic_lazy_12_clean_with_interventions_corrected.npz")
            out_dir = ep_cfg.get("output_dir", f"results/plots/early_prediction/{cfg.experiment_id}")
            
            target_models = ep_cfg.get("target_models", ["lstm_no_v", "lstm_with_v", "transformer_no_v", "transformer_with_v"])
            if isinstance(target_models, str):
                target_models = [m.strip() for m in target_models.split(",")]
                
            python_exe = get_python_executable()
            slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
            slurm_dir.mkdir(parents=True, exist_ok=True)
            
            for m_target in target_models:
                print(f"\n--> Setting up Optuna Study for architecture target: [{m_target}]")
                tune_args = [
                    "--n-trials", str(n_trials),
                    "--model-target", str(m_target),
                    "--checkpoint", str(ckpt),
                    "--dataset-path", str(dataset_path),
                    "--out-dir", str(out_dir)
                ]
                if local_val:
                    cmd = [python_exe, "-u", "src/early_prediction/tune_optuna.py"] + tune_args
                    print(f"Executing local: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                else:
                    slurm_script_path = slurm_dir / f"tune_pred_{m_target}.slurm"
                    cmd_str = " ".join([f'"{arg}"' if " " in arg else arg for arg in tune_args])
                    script_content = f"""#!/bin/bash
#SBATCH --job-name=tune_{m_target}_{cfg.experiment_id}
#SBATCH --partition=rtx4060ti8g
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --output=results/logs/slurm/{cfg.group}/{cfg.experiment_id}/tune_{m_target}_%j.out
#SBATCH --error=results/logs/slurm/{cfg.group}/{cfg.experiment_id}/tune_{m_target}_%j.err

echo "=== Sepsis Early Prediction Optuna Search [{m_target}] Start ==="
echo "Node: $(hostname)"
date

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/nsfr:$PROJECT_ROOT/src/nudge:$PROJECT_ROOT/src/neumann:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

mkdir -p {out_dir}
mkdir -p results/logs

$PROJECT_ROOT/venv/bin/python3 -u src/early_prediction/tune_optuna.py \\
    {cmd_str}

echo "=== Sepsis Early Prediction Optuna Search [{m_target}] End ==="
date
"""
                    with open(slurm_script_path, "w") as f:
                        f.write(script_content)
                    print(f"Submitting Early Prediction Optuna SLURM Job [{m_target}]: {slurm_script_path}")
                    res = subprocess.run(["sbatch", str(slurm_script_path)], capture_output=True, text=True)
                    print(res.stdout)
                    if res.stderr:
                        print(res.stderr)
            sys.exit(0)

    # Track the best trial ID for each online method to use its dataset later (Local only)
    best_online_trial_ids = {}

    # 1. Online Training Phases
    if not args.no_online:
        for agent_config in online_list:
            agent_name_internal = agent_config.replace("/", "_")
            study_name = f"{cfg.experiment_id}_{agent_name_internal}"
            
            if local_val:
                dataset_path = f"in/datasets/{cfg.group}/{args.experiment}/{agent_name_internal}"
            else:
                if is_sweep:
                    dataset_path = f"in/datasets/{cfg.experiment_id}/{agent_name_internal}"
                else:
                    dataset_path = f"in/datasets/{cfg.group}/{cfg.experiment_id}/{agent_name_internal}"
            
            # Check if dataset already exists to skip training
            has_pkl = False
            if os.path.exists(dataset_path):
                if not local_val:
                    for root, dirs, files in os.walk(dataset_path):
                        if any(f.endswith(".pkl") for f in files):
                            has_pkl = True
                            break
                else:
                    has_pkl = any(f.endswith(".pkl") for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)))
                
            if not has_pkl:
                found_path = find_dataset_globally(agent_name_internal)
                if found_path:
                    print(f"Dataset found globally at {found_path}. Symlinking to expected path {dataset_path}...")
                    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                    if os.path.lexists(dataset_path):
                        if os.path.isdir(dataset_path) and not os.path.islink(dataset_path):
                            import shutil
                            shutil.rmtree(dataset_path)
                        else:
                            os.unlink(dataset_path)
                    # Use relative symlink to ensure compatibility on cluster nodes
                    rel_source = os.path.relpath(os.path.abspath(found_path), start=os.path.dirname(os.path.abspath(dataset_path)))
                    os.symlink(rel_source, dataset_path)
                    has_pkl = True

            if has_pkl:
                 print(f"Dataset already exists at {dataset_path}. Skipping online training.")
                 if local_val:
                     best_online_trial_ids[agent_config] = "0"
                 else:
                     online_job_ids[agent_config] = None
                 continue

            if local_val:
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
                    
                run_experiment(overrides)
                
                # After training, find the best trial ID if we were sweeping
                if is_sweep:
                    best_id = get_best_trial_id(storage_url, study_name)
                    best_online_trial_ids[agent_config] = best_id
                    print(f"Best trial for {agent_config} was ID: {best_id}")
                else:
                    best_online_trial_ids[agent_config] = "0"
            else:
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
                    overrides_slurm.append(f"++hydra.sweeper.study_name={study_name}")
                    delete_optuna_study(storage_url, study_name)
                    create_optuna_study(storage_url, study_name)
                else:
                    overrides_slurm.append(f"++dataset_path={dataset_path}")
                overrides_slurm += sanitized_extra_args
                
                script_content = generate_sbatch_script(
                    job_name, overrides_slurm, log_dir=str(log_dir),
                    partition=args.partition, gpus=args.gpus, cores=args.cores, nodes=args.nodes
                )
                job_id = submit_sbatch(script_content)
                if job_id:
                    job_ids.append(job_id)
                    online_job_ids[agent_config] = job_id
    else:
        print("\n=== Skipping Online Training Phase ===")

    # 2. Offline Training Phases (Many-to-Many)
    eval_job_ids = []
    eval_commands = []
    if not args.no_offline:
        for dataset_id in dataset_list:
            dataset_name_internal = dataset_id.replace("/", "_")
            is_online = dataset_id in online_list
            dependency_job_id = online_job_ids.get(dataset_id) if not local_val else None
            
            if local_val:
                best_id = best_online_trial_ids.get(dataset_id, "0")
                dataset_path = Path("in/datasets") / cfg.group / args.experiment / dataset_name_internal / best_id
                if not dataset_path.exists():
                    alt_path = Path("in/datasets") / cfg.group / args.experiment / dataset_name_internal
                    if alt_path.exists() and any(alt_path.glob("*.pkl")):
                        dataset_path = alt_path
                    else:
                        group_shared_path = Path("in/datasets") / cfg.group / dataset_name_internal
                        if group_shared_path.exists() and any(group_shared_path.glob("*.pkl")):
                            dataset_path = group_shared_path
                        else:
                            global_match = find_dataset_globally(dataset_name_internal)
                            if global_match:
                                dataset_path = Path(global_match)
                            else:
                                print(f"Error: Dataset '{dataset_id}' not found at {dataset_path} or globally.")
                                sys.exit(1)
                print(f"Using dataset from: {dataset_path}")

            for agent_config in offline_list:
                agent_name_internal = agent_config.replace("/", "_")
                study_name = f"{args.experiment}_{agent_name_internal}_{dataset_name_internal}"
                
                if local_val:
                    print(f"\n=== Phase: Offline Training ({agent_config}) on Dataset ({dataset_id}) ===")
                    dataset_path_override = any("mode.dataset_path=" in arg for arg in sanitized_extra_args)
                    overrides = [
                        f"+experiment={args.experiment}",
                        f"++local=true",
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
                else:
                    print(f"\n=== Preparing Slurm Job: Offline Training ({agent_config}) on Dataset ({dataset_id}) ===")
                    job_name = f"{agent_name_internal}_{dataset_name_internal}_{cfg.experiment_id}"
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
                        delete_optuna_study(storage_url, study_name)
                        create_optuna_study(storage_url, study_name)
                    
                    if is_online and is_sweep:
                        storage_url_slurm = storage_url if storage_url else f"sqlite:///results/optuna/optuna.db"
                        study_name_slurm = f"{cfg.experiment_id}_{dataset_name_internal}"
                        best_id_cmd = f"BEST_ID=$($PROJECT_ROOT/venv/bin/python3 -c \"import sys; sys.path.append('$PROJECT_ROOT'); from run_pipeline import get_best_trial_id; print(get_best_trial_id('{storage_url_slurm}', '{study_name_slurm}'))\")"
                        dataset_path_cmd = f"D_PATH=in/datasets/{cfg.experiment_id}/{dataset_name_internal}/$BEST_ID"
                        
                        cmd_args = overrides_slurm + sanitized_extra_args
                        import shlex
                        train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
                        
                        script_content = f"#!/bin/bash\n"
                        script_content += f"#SBATCH --job-name={job_name}\n"
                        script_content += f"#SBATCH --partition={args.partition}\n"
                        script_content += f"#SBATCH --time=04:00:00\n"
                        script_content += f"#SBATCH --ntasks-per-node={args.cores}\n"
                        script_content += f"#SBATCH --nodes={args.nodes}\n"
                        script_content += f"#SBATCH --output={log_dir}/%x_%j.out\n"
                        script_content += f"#SBATCH --error={log_dir}/%x_%j.err\n"
                        script_content += f"#SBATCH --mail-type=END,FAIL\n"
                        script_content += f"#SBATCH --mail-user=cegbert@ncsu.edu\n"
                        if dependency_job_id:
                            script_content += f"#SBATCH --dependency=afterok:{dependency_job_id}\n"
                        script_content += f"\n"
                        script_content += f"export PROJECT_ROOT={os.getcwd()}\n"
                        script_content += f"export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH\n"
                        script_content += f"{best_id_cmd}\n"
                        script_content += f"{dataset_path_cmd}\n"
                        script_content += f"if [ ! -d \"$D_PATH\" ] || [ -z \"$(ls $D_PATH/*.pkl 2>/dev/null)\" ]; then\n"
                        script_content += f"    echo \"Best trial dataset not found at $D_PATH. Falling back to parent directory.\"\n"
                        script_content += f"    D_PATH=in/datasets/{cfg.experiment_id}/{dataset_name_internal}\n"
                        script_content += f"    if [ ! -d \"$D_PATH\" ] || [ -z \"$(ls $D_PATH/*.pkl 2>/dev/null)\" ]; then\n"
                        script_content += f"        D_PATH=in/datasets/{cfg.group}/{cfg.experiment_id}/{dataset_name_internal}\n"
                        script_content += f"    fi\n"
                        script_content += f"fi\n"
                        script_content += f"echo \"Using dataset: $D_PATH\"\n"
                        script_content += f"$PROJECT_ROOT/venv/bin/python3 {train_cmd} ++mode.dataset_path=$D_PATH\n"
                    else:
                        dataset_path = Path("in/datasets") / cfg.group / cfg.experiment_id / dataset_name_internal
                        if not (dataset_path.exists() and any(dataset_path.glob("*.pkl"))):
                            group_shared_path = Path("in/datasets") / cfg.group / dataset_name_internal
                            if group_shared_path.exists() and any(group_shared_path.glob("*.pkl")):
                                dataset_path = group_shared_path
                            else:
                                global_match = find_dataset_globally(dataset_name_internal)
                                if global_match:
                                    dataset_path = Path(global_match)
                        overrides_slurm.append(f"++mode.dataset_path={dataset_path}")
                        overrides_slurm += sanitized_extra_args
                        script_content = generate_sbatch_script(
                            job_name, overrides_slurm, log_dir=str(log_dir),
                            partition=args.partition, gpus=args.gpus, cores=args.cores, nodes=args.nodes,
                            dependency=dependency_job_id
                        )
                    
                    job_id = submit_sbatch(script_content)
                    if job_id:
                        job_ids.append(job_id)
                        
                        # Pipeline Integration: MIMIC early prediction evaluator for Slurm
                        if cfg.env.name == "mimic":
                            if is_sweep:
                                storage_url_slurm = storage_url if storage_url else f"sqlite:///results/optuna/optuna.db"
                                if is_online:
                                    study_name_slurm = f"{cfg.experiment_id}_{dataset_name_internal}"
                                else:
                                    study_name_slurm = f"{cfg.experiment_id}_{agent_name_internal}_{dataset_name_internal}"
                                best_id_cmd = (
                                    f"BEST_ID=$($PROJECT_ROOT/venv/bin/python3 -c \"import sys; sys.path.append('$PROJECT_ROOT'); from run_pipeline import get_best_trial_id; print(get_best_trial_id('{storage_url_slurm}', '{study_name_slurm}'))\")\n"
                                )
                                if args.remake:
                                    eval_cmd = (
                                        best_id_cmd +
                                        f"CKPT_DIR=results/checkpoints/{cfg.group}/{cfg.experiment_id}/{agent_name_internal}/$BEST_ID\n"
                                        f"if [ -d \"$CKPT_DIR\" ]; then\n"
                                        f"    echo \"Running evaluation on all checkpoints under $CKPT_DIR (--remake)\"\n"
                                        f"    $PROJECT_ROOT/venv/bin/python3 src/early_prediction/eval.py --checkpoint $CKPT_DIR --remake\n"
                                        f"else\n"
                                        f"    echo \"Checkpoint dir not found at $CKPT_DIR\"\n"
                                        f"fi"
                                    )
                                else:
                                    eval_cmd = (
                                        best_id_cmd +
                                        f"CKPT_PATH=results/checkpoints/{cfg.group}/{cfg.experiment_id}/{agent_name_internal}/$BEST_ID/best_model.ckpt\n"
                                        f"if [ -f \"$CKPT_PATH\" ]; then\n"
                                        f"    echo \"Running evaluation on $CKPT_PATH\"\n"
                                        f"    $PROJECT_ROOT/venv/bin/python3 src/early_prediction/eval.py --checkpoint $CKPT_PATH\n"
                                        f"else\n"
                                        f"    echo \"Checkpoint not found at $CKPT_PATH\"\n"
                                        f"fi"
                                    )
                            else:
                                ckpt_path = f"results/checkpoints/{cfg.group}/{cfg.experiment_id}/{agent_name_internal}/0/best_model.ckpt"
                                ckpt_dir = f"results/checkpoints/{cfg.group}/{cfg.experiment_id}/{agent_name_internal}/0"
                                if args.remake:
                                    eval_cmd = f"$PROJECT_ROOT/venv/bin/python3 src/early_prediction/eval.py --checkpoint {ckpt_dir} --remake"
                                else:
                                    eval_cmd = f"$PROJECT_ROOT/venv/bin/python3 src/early_prediction/eval.py --checkpoint {ckpt_path}"
                                
                            eval_commands.append(eval_cmd)
    else:
        print("\n=== Skipping Offline Training Phase ===")

    # Add evaluation job IDs to dependency list so that final plotting waits for them
    job_ids.extend(eval_job_ids)

    # 3. Pipeline Integration: Local early prediction evaluation
    if local_val and cfg.env.name == "mimic":
        print("\n=== Phase: Local Early Prediction Evaluation ===")
        ckpt_dir_root = Path("results/checkpoints") / cfg.group / cfg.experiment_id
        if ckpt_dir_root.exists():
            if args.remake:
                # Pass parent directories of best_model files to run them all at once
                checkpoint_dirs = set()
                for cp_path in ckpt_dir_root.glob("**/best_model*.ckpt"):
                    checkpoint_dirs.add(cp_path.parent)
                if checkpoint_dirs:
                    for cp_dir in sorted(checkpoint_dirs):
                        run_early_prediction_eval(cp_dir, remake=True)
                else:
                    print(f"Warning: No best_model*.ckpt files found under {ckpt_dir_root} for evaluation.")
            else:
                checkpoint_files = []
                for path_dir in ckpt_dir_root.glob("**/"):
                    if path_dir.is_dir():
                        candidates = list(path_dir.glob("best_model*.ckpt"))
                        if candidates:
                            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                            checkpoint_files.append(candidates[0])
                if checkpoint_files:
                    for ckpt_file in checkpoint_files:
                        run_early_prediction_eval(ckpt_file)
                else:
                    print(f"Warning: No best_model*.ckpt files found under {ckpt_dir_root} for evaluation.")
        else:
            print(f"Warning: Checkpoint root directory {ckpt_dir_root} does not exist.")

    # 4. Plotting
    if not args.no_plot:
        if local_val:
            run_plotting(args.experiment, style=args.plot_style)
        else:
            if job_ids:
                actual_exp_id = cfg.experiment_id
                print(f"\n=== Preparing Final Job: Evaluation and Plotting ({actual_exp_id}) ===")

                job_name = f"final_{actual_exp_id}"
                
                all_dependencies = ":".join(jid for jid in job_ids if jid != "99999")
                
                project_root = os.getcwd()
                plot_cmd = f"{project_root}/venv/bin/python3 plot/manager.py {actual_exp_id}"
                if args.plot_style:
                    plot_cmd += f" --style {args.plot_style}"
                    
                final_script = f"#!/bin/bash\n"
                final_script += f"#SBATCH --job-name={job_name}\n"
                final_script += f"#SBATCH --partition={args.partition}\n"
                final_script += f"#SBATCH --ntasks-per-node=1\n"
                final_script += f"#SBATCH --nodes=1\n"
                final_script += f"#SBATCH --output={log_dir}/%x_%j.out\n"
                final_script += f"#SBATCH --error={log_dir}/%x_%j.err\n"
                final_script += f"#SBATCH --mail-type=END,FAIL\n"
                final_script += f"#SBATCH --mail-user=cegbert@ncsu.edu\n"

                if all_dependencies:
                    final_script += f"#SBATCH --dependency=afterany:{all_dependencies}\n"
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
            for m in online_methods + offline_methods:
                cl = clean_label(m.replace("/", "_"))
                if cl not in ran_methods:
                    ran_methods.append(cl)

            print(f"\n" + "="*40)
            print(f"SUBMISSION SUMMARY")
            print(f"Experiment:    {cfg.group}/{cfg.experiment_id}")
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
