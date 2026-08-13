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


def get_best_trial_id(storage_url, study_name):
    """Queries the Optuna database to find the best trial ID for a given study.
    
    Uses the Optuna Python API instead of raw SQL for robustness across
    schema versions.
    """
    if not storage_url:
        return "0"
    
    try:
        import optuna
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        return str(study.best_trial.number)
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
