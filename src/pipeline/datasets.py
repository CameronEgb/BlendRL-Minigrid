"""Dataset resolution and experiment execution utilities.

Provides functions for finding datasets, running experiments, and triggering plots.
"""
import os
import subprocess
from pathlib import Path

from src.pipeline.config import get_python_executable


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


def ensure_online_dataset_path(group: str, experiment_id: str, agent_name_internal: str, is_sweep: bool = False):
    """Determine expected online dataset path and ensure directory / symlink exists if dataset is available globally.
    
    Returns:
        tuple[str, bool]: (dataset_path, has_existing_dataset)
    """
    if is_sweep:
        dataset_path = f"in/datasets/{experiment_id}/{agent_name_internal}"
    else:
        dataset_path = f"in/datasets/{group}/{experiment_id}/{agent_name_internal}"
        
    has_pkl = False
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            if any(f.endswith(".pkl") for f in files):
                has_pkl = True
                break
                
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

    return dataset_path, has_pkl


def resolve_dataset_path(dataset_id: str, group: str = "", experiment_id: str = "", yaml_ds_path: str = None) -> Path:
    """Robustly resolve the filesystem path for an offline dataset.
    
    Checks in order:
    1. in/datasets/[group]/[experiment_id]/[dataset_name]
    2. in/datasets/[group]/[dataset_name]
    3. in/datasets/[dataset_name]
    4. in/datasets/[dataset_id]
    5. yaml_ds_path (if explicitly configured and valid)
    6. Shallowest global match under in/datasets/
    """
    dataset_name_internal = dataset_id.replace("/", "_")
    
    candidates = []
    if group and experiment_id:
        candidates.append(Path("in/datasets") / group / experiment_id / dataset_name_internal)
    if group:
        candidates.append(Path("in/datasets") / group / dataset_name_internal)
    candidates.extend([
        Path("in/datasets") / dataset_name_internal,
        Path("in/datasets") / dataset_id,
    ])
    if yaml_ds_path:
        candidates.append(Path(yaml_ds_path))
        
    for cand in candidates:
        if cand.exists() and any(cand.glob("*.pkl")):
            return cand
        if cand.with_suffix(".npz").exists() or (cand.parent / f"{cand.name}.npz").exists():
            return cand
            
    # Try global lookup
    global_match = find_dataset_globally(dataset_name_internal)
    if global_match:
        return Path(global_match)
        
    if yaml_ds_path and Path(yaml_ds_path).exists():
        return Path(yaml_ds_path)
        
    if group:
        return Path("in/datasets") / group / dataset_name_internal
        
    return Path("in/datasets") / dataset_name_internal


def resolve_mimic_npz_path(filename_or_path: str = None) -> Path:
    """Robustly resolve the filesystem path for a MIMIC NPZ dataset file.
    
    Raises FileNotFoundError if the file cannot be located.
    """
    if filename_or_path:
        p = Path(filename_or_path)
        if p.exists() and p.is_file():
            return p.resolve()
        filename = p.name
    else:
        filename = os.environ.get("MIMIC_DATASET_NAME", "")
        if not filename:
            raise ValueError(
                "No MIMIC dataset specified to resolve_mimic_npz_path. "
                "Please pass a dataset path or set env.dataset_name in the experiment configuration."
            )

    candidate_dirs = [
        os.environ.get("MIMIC_DATASET_DIR", ""),
        Path("in/datasets/mimic"),
        Path("in/datasets/MIMIC 2"),
        Path("in/datasets"),
        Path.home() / "Documents/NCSU/Research/datasets/MIMIC 2",
        Path.home() / "Offline-BlendRL/in/datasets/mimic",
        Path.home() / "Offline-BlendRL/in/datasets",
        Path("/hpc/home/cegbert1/Offline-BlendRL/in/datasets/mimic"),
        Path("/hpc/home/cegbert1/Offline-BlendRL/in/datasets"),
    ]
    candidate_dirs = [Path(d).resolve() for d in candidate_dirs if d and (isinstance(d, Path) or len(str(d)) > 0)]

    for c_dir in candidate_dirs:
        if not c_dir.exists():
            continue
        cand = c_dir / filename
        if cand.exists() and cand.is_file():
            return cand.resolve()

    searched = [str(d) for d in candidate_dirs if d.exists()]
    raise FileNotFoundError(
        f"MIMIC dataset '{filename}' not found. Searched existing directories: {searched}"
    )


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


def run_plotting(experiment, style=None, base_experiment=None):
    """Run plot/manager.py for the given experiment."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".") + ":" + os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
    
    venv_python = get_python_executable()
    cmd = [venv_python, "plot/manager.py", str(experiment)]
    if base_experiment and str(base_experiment) != str(experiment):
        cmd.extend(["--experiment", str(base_experiment)])
    if style:
        cmd.extend(["--style", str(style)])
        
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
