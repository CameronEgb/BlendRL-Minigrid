"""Dataset resolution and experiment execution utilities.

Provides functions for finding datasets, running experiments, and triggering plots.
"""
import functools
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from src.pipeline.runtime import get_python_executable


def fast_purge_dir(path: Path):
    """Instantly remove a directory on network storage (NFS) via atomic rename + background purge."""
    if not path.exists():
        return
    trash_path = path.parent / f".trash_{path.name}_{uuid.uuid4().hex[:8]}"
    try:
        path.rename(trash_path)
        threading.Thread(target=shutil.rmtree, args=(trash_path, True), daemon=True).start()
    except Exception:
        shutil.rmtree(path, ignore_errors=True)


def ensure_online_dataset_path(group: str, experiment_id: str, agent_name_internal: str, is_sweep: bool = False):
    """Determine expected online dataset path and ensure it exists.
    
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

    return dataset_path, has_pkl


@functools.lru_cache(maxsize=128)
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
    alt_ids = [dataset_id, dataset_name_internal]
    if not dataset_id.endswith("(w)"):
        alt_ids.extend([f"{dataset_id}(w)", dataset_id.replace("_w", "(w)")])
    
    candidates = []
    for aid in alt_ids:
        aid_clean = aid.replace("/", "_")
        if group and experiment_id:
            candidates.append(Path("in/datasets") / group / experiment_id / aid_clean)
            candidates.append(Path("in/datasets") / group / experiment_id / aid)
        if group:
            candidates.append(Path("in/datasets") / group / "per_problem" / aid / "cql")
            candidates.append(Path("in/datasets") / group / "per_problem" / aid_clean / "cql")
            candidates.append(Path("in/datasets") / group / "per_problem" / aid)
            candidates.append(Path("in/datasets") / group / "per_problem" / aid_clean)
            candidates.append(Path("in/datasets") / group / aid / "cql")
            candidates.append(Path("in/datasets") / group / aid_clean / "cql")
            candidates.append(Path("in/datasets") / group / aid_clean)
            candidates.append(Path("in/datasets") / group / aid)
        candidates.extend([
            Path("in/datasets") / "per_problem" / aid / "cql",
            Path("in/datasets") / aid_clean,
            Path("in/datasets") / aid,
        ])
    if yaml_ds_path:
        candidates.append(Path(yaml_ds_path))
        
    for cand in candidates:
        if cand.exists() and any(cand.glob("*.pkl")):
            return cand
        if cand.exists() and (cand / "cql").exists() and any((cand / "cql").glob("*.pkl")):
            return cand / "cql"
        if cand.with_suffix(".npz").exists() or (cand.parent / f"{cand.name}.npz").exists():
            return cand
            
    if yaml_ds_path and Path(yaml_ds_path).exists():
        return Path(yaml_ds_path)
        
    if group:
        return Path("in/datasets") / group / dataset_name_internal
        
    return Path("in/datasets") / dataset_name_internal


def resolve_mimic_npz_path(filename_or_path: str = None, site_cfg=None) -> Path:
    """Robustly resolve the filesystem path for a MIMIC NPZ dataset file.
    
    Uses site_cfg.dataset_search_dirs for cluster-specific paths instead of
    hardcoding personal directories.
    
    Raises FileNotFoundError if the file cannot be located.
    """
    from src.pipeline.runtime import PROJECT_ROOT

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
        PROJECT_ROOT / "in/datasets/mimic",
        PROJECT_ROOT / "in/datasets",
    ]
    # Add site-specific search dirs (replaces hardcoded /hpc/home/cegbert1/... paths)
    if site_cfg:
        for d in getattr(site_cfg, "dataset_search_dirs", []) or []:
            candidate_dirs.append(Path(d))
    candidate_dirs = [Path(d).resolve() for d in candidate_dirs if d and str(d).strip()]

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



def run_experiment(overrides, site_cfg=None):
    """Run src/train.py as a subprocess with the given Hydra overrides."""
    from src.pipeline.runtime import get_subprocess_env
    env = get_subprocess_env(site_cfg)
    
    sanitized_overrides = []
    for arg in overrides:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if ("(" in v or ")" in v) and not (v.startswith("'") or v.startswith('"') or v.startswith("[") or v.startswith("{")):
                arg = f"{k}='{v}'"
        sanitized_overrides.append(arg)

    venv_python = get_python_executable()
    cmd = [venv_python, "src/train.py"] + sanitized_overrides
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def run_plotting(experiment, style=None, base_experiment=None, site_cfg=None):
    """Run plot/manager.py for the given experiment."""
    from src.pipeline.runtime import get_subprocess_env
    env = get_subprocess_env(site_cfg)
    
    venv_python = get_python_executable()
    cmd = [venv_python, "plot/manager.py", str(experiment)]
    if base_experiment and str(base_experiment) != str(experiment):
        cmd.extend(["--experiment", str(base_experiment)])
    if style:
        cmd.extend(["--style", str(style)])
        
    print(f"\n=== Auto-Generating Modular Plots for experiment: {experiment} ===")
    subprocess.run(cmd, check=True, env=env)


