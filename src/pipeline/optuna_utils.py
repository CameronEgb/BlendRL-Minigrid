"""Optuna-related utilities for the pipeline.

Provides functions for querying best trials, managing studies,
and launching the Optuna dashboard.
"""
import os
import subprocess
import sys
import time
import threading
import webbrowser


DEFAULT_OPTUNA_DB_URL = "sqlite:///results/optuna/optuna.db"


from src.pipeline.runtime import get_python_executable


def is_valid_storage_url(storage_url) -> bool:
    """Checks whether the given storage URL is non-empty and not in-memory (None/null)."""
    if not storage_url:
        return False
    s = str(storage_url).strip().lower()
    return s not in ("none", "null", "", "false")


def get_best_trial_id(storage_url, study_name):
    """Queries the Optuna database to find the best trial ID for a given study.
    
    Uses the Optuna Python API instead of raw SQL for robustness across
    schema versions.
    """
    if not is_valid_storage_url(storage_url):
        return "0"
    
    try:
        import optuna
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            return str(study.best_trial.number)
        except Exception:
            all_studies = optuna.get_all_study_summaries(storage=storage_url)
            matches = [s for s in all_studies if s.study_name == study_name or s.study_name.startswith(f"{study_name}_v")]
            if matches:
                matches.sort(key=lambda s: s.study_name, reverse=True)
                target_study = optuna.load_study(study_name=matches[0].study_name, storage=storage_url)
                return str(target_study.best_trial.number)
            raise
    except Exception as e:
        print(f"Warning: Could not query best trial from Optuna: {e}")
        return "0"


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
    except Exception:
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
            except Exception:
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


def get_optuna_storage(storage_url: str):
    """Safely return an Optuna RDBStorage object with timeout and clean schema handling."""
    if not storage_url:
        return None
    import optuna
    if storage_url.startswith("sqlite:///"):
        import sqlite3
        db_raw = storage_url.replace("sqlite:///", "").split("?")[0]
        if db_raw and os.path.dirname(db_raw):
            os.makedirs(os.path.dirname(os.path.abspath(db_raw)), exist_ok=True)
        # If DB file exists but is 0 bytes (empty file created by touch/connect),
        # remove it so Optuna creates the full table schema cleanly without AssertionError
        if os.path.exists(db_raw) and os.path.getsize(db_raw) == 0:
            try:
                os.remove(db_raw)
            except OSError:
                pass
        # Pre-configure SQLite database with WAL journal mode and busy timeout via Python
        try:
            conn = sqlite3.connect(db_raw, timeout=120.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=120000;")
            conn.close()
        except Exception:
            pass
        return optuna.storages.RDBStorage(
            url=f"sqlite:///{db_raw}",
            engine_kwargs={"connect_args": {"timeout": 120}}
        )
    return storage_url


def get_next_study_name(group: str, experiment_id: str, agent_name: str, storage_url: str = None) -> str:
    """Generate clean [experiment]_[method]_v[number] study name using existing run versions.
    
    Uses filesystem inspection to avoid database locking during batch submission.
    """
    base_prefix = f"{experiment_id}_{agent_name}"
    from pathlib import Path
    
    log_dir = Path("results/logs") / group / experiment_id / agent_name
    ckpt_dir = Path("results/checkpoints") / group / experiment_id / agent_name
    
    existing_versions = [0]
    for d in [log_dir, ckpt_dir]:
        if d.exists():
            for v_path in d.glob("version_*"):
                if v_path.is_dir():
                    try:
                        existing_versions.append(int(v_path.name.split("_")[-1]) + 1)
                    except ValueError:
                        pass
    version = max(existing_versions)
    return f"{base_prefix}_v{version}"


def delete_optuna_study(storage_url, study_name):
    """Deletes an existing study from the Optuna database to start fresh."""
    if not is_valid_storage_url(storage_url):
        return
    try:
        import optuna
        storage = get_optuna_storage(storage_url)
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f"Reset existing Optuna study: {study_name}")
    except Exception:
        pass


def create_optuna_study(storage_url, study_name, direction="minimize"):
    """Pre-creates/initializes an Optuna study to avoid schema initialization race conditions on cluster nodes."""
    if not is_valid_storage_url(storage_url):
        return
    try:
        import optuna
        storage = get_optuna_storage(storage_url)
        optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction=direction)
        print(f"Pre-initialized Optuna study: {study_name}")
    except Exception as e:
        print(f"Warning: Failed to pre-initialize Optuna study '{study_name}': {e}")


def find_best_trial_from_logs(group: str, experiment_id: str, agent_name: str, direction: str = "minimize"):
    """Scan CSV logs to find the winning trial number and score for in-memory sweeps."""
    import pandas as pd
    from pathlib import Path
    log_dir = Path("results/logs") / group / experiment_id / agent_name
    best_id = "0"
    best_val = float("inf") if direction == "minimize" else float("-inf")
    
    if not log_dir.exists():
        return best_id, None
        
    for v_dir in sorted(log_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0):
        metrics_file = v_dir / "metrics.csv"
        if not metrics_file.exists():
            continue
        try:
            df = pd.read_csv(metrics_file)
            trial_num = v_dir.name.split("_")[-1]
            val = None
            for col in ["val/loss", "val/robust_loss", "losses/bellman_loss", "losses/total_loss", "eval/reward"]:
                if col in df.columns:
                    valid_series = df[col].dropna()
                    if len(valid_series) > 0:
                        val = float(valid_series.iloc[-1])
                        break
            if val is not None:
                if (direction == "minimize" and val < best_val) or (direction == "maximize" and val > best_val):
                    best_val = val
                    best_id = trial_num
        except Exception:
            pass
            
    return best_id, (best_val if best_val != float("inf") and best_val != float("-inf") else None)


def promote_best_trial_checkpoint(group: str, experiment_id: str, agent_name: str, storage_url: str = None, study_name: str = None):
    """After an Optuna sweep, queries the winning trial, copies its checkpoint
    and hyperparameters to the main agent checkpoint root, and writes a summary."""
    import shutil
    import json
    import yaml
    from pathlib import Path
    
    ckpt_root = Path("results/checkpoints") / group / experiment_id / agent_name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    target_ckpt_path = ckpt_root / "best_model.ckpt"
    
    best_info = {"study_name": study_name}
    best_id = "0"
    
    if is_valid_storage_url(storage_url):
        best_id = get_best_trial_id(storage_url, study_name)
        try:
            import optuna
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            best_info["best_value"] = float(study.best_trial.value) if study.best_trial.value is not None else None
            best_info["best_params"] = study.best_trial.params
            best_info["direction"] = str(study.direction.name)
            
            # Write best params yaml
            best_params_yaml = ckpt_root / "best_params.yaml"
            with open(best_params_yaml, "w") as f:
                yaml.dump(study.best_trial.params, f, default_flow_style=False)
        except Exception as e:
            print(f"Notice: Could not load full Optuna study details: {e}")
    else:
        # In-memory storage: Find best trial from CSV logs
        best_id, best_val = find_best_trial_from_logs(group, experiment_id, agent_name)
        best_info["best_value"] = best_val
        best_info["direction"] = "minimize"
        
        # Copy hparams if available
        hparams_src = Path("results/logs") / group / experiment_id / agent_name / f"version_{best_id}" / "hparams.yaml"
        if hparams_src.exists():
            shutil.copy2(hparams_src, ckpt_root / "best_params.yaml")
            
    best_info["best_trial_id"] = best_id
    trial_ckpt_path = ckpt_root / best_id / "best_model.ckpt"
    
    if trial_ckpt_path.exists():
        shutil.copy2(trial_ckpt_path, target_ckpt_path)
        # Also save explicitly named checkpoint in both local folder and parent experiment root
        named_ckpt_path = ckpt_root / f"{agent_name}.ckpt"
        shutil.copy2(trial_ckpt_path, named_ckpt_path)
        parent_ckpt_root = Path("results/checkpoints") / group / experiment_id
        shutil.copy2(trial_ckpt_path, parent_ckpt_root / f"{agent_name}.ckpt")
        print(f"\n[Optuna Winner] Promoted Best Trial #{best_id} -> {named_ckpt_path.name}")
        if "best_value" in best_info and best_info["best_value"] is not None:
            print(f"[Optuna Winner] Best Score ({best_info.get('direction', 'metric')}): {best_info['best_value']:.4f}")
    else:
        all_ckpts = list(ckpt_root.rglob("best_model*.ckpt"))
        # Exclude target if it's in the list
        all_ckpts = [c for c in all_ckpts if c.resolve() != target_ckpt_path.resolve()]
        if all_ckpts:
            shutil.copy2(all_ckpts[0], target_ckpt_path)
            named_ckpt_path = ckpt_root / f"{agent_name}.ckpt"
            shutil.copy2(all_ckpts[0], named_ckpt_path)
            parent_ckpt_root = Path("results/checkpoints") / group / experiment_id
            shutil.copy2(all_ckpts[0], parent_ckpt_root / f"{agent_name}.ckpt")
            print(f"\n[Optuna Fallback] Promoted checkpoint -> {named_ckpt_path.name}")
            
    with open(ckpt_root / "best_trial_summary.json", "w") as f:
        json.dump(best_info, f, indent=2)
        
    return best_id
