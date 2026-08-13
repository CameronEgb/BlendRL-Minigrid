"""Dataset resolution and experiment execution utilities.

Provides functions for finding datasets, running experiments, and triggering plots.
"""
import os
import subprocess
from pathlib import Path

from src.pipeline.optuna_utils import get_python_executable


def find_dataset_globally(agent_name_internal):
    """Search for existing datasets under in/datasets/ by agent name.
    
    Returns the shallowest matching path, or None if not found.
    """
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


def run_experiment(overrides):
    """Run src/train.py as a subprocess with the given Hydra overrides."""
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
    """Run plot/manager.py for the given experiment."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".") + ":" + os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
    
    venv_python = get_python_executable()
    cmd = [venv_python, "plot/manager.py", experiment]
        
    print(f"\n=== Auto-Generating Modular Plots for experiment: {experiment} ===")
    subprocess.run(cmd, check=True, env=env)


def run_early_prediction_eval(checkpoint_path, remake=False):
    """Run early prediction evaluation for a given checkpoint."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
    
    venv_python = get_python_executable()
    cmd = [
        venv_python, "src/early_prediction/eval.py",
        "--checkpoint", str(checkpoint_path),
        "--ep-ckpt-root", "results/checkpoints/early_prediction",
    ]
    if remake:
        cmd.append("--remake")
    print(f"\n=== Running Early Prediction Evaluation for checkpoint: {checkpoint_path} ===")
    subprocess.run(cmd, check=True, env=env)
