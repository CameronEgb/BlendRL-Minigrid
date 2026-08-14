"""Pre-flight configuration validation module."""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

from src.methods.registry import list_registered_agents, auto_discover
from src.pipeline.datasets import resolve_dataset_path
from src.pipeline.config import parse_method_list


def validate_experiment_config(cfg: Any, experiment_name: str, is_sweep: bool = False) -> List[str]:
    """Validate experiment configuration and return list of non-fatal notices.
    Raises ValueError on fatal configuration errors."""
    auto_discover()
    issues = []
    fatal_errors = []
    
    env_name = getattr(cfg.env, "name", "unknown") if hasattr(cfg, "env") else "unknown"
    is_offline_only = getattr(cfg.env, "offline_only", False) or env_name in ["mimic", "pyrenees"]
    mode_type = getattr(cfg.mode, "type", "online") if hasattr(cfg, "mode") else "online"
    
    # 1. Check intervals_count on offline-only datasets
    if is_offline_only:
        # Check if the user explicitly set intervals_count > 1 in the experiment file
        exp_path = Path("in/config/experiment") / f"{experiment_name}.yaml"
        if not exp_path.exists():
            matches = list(Path("in/config/experiment").glob(f"**/{experiment_name}.yaml"))
            if matches: exp_path = matches[0]
        if exp_path.exists():
            import yaml
            with open(exp_path) as f:
                raw_exp = yaml.safe_load(f) or {}
                if raw_exp.get("intervals_count", 1) > 1:
                    fatal_errors.append(
                        f"Config '{experiment_name}' explicitly sets intervals_count={raw_exp['intervals_count']} for offline-only environment '{env_name}'. "
                        f"intervals_count MUST be 1 for static offline datasets."
                    )
            
    # 2. Validate agent registrations
    registered = set(list_registered_agents())
    
    online_methods = parse_method_list(cfg.get("online_methods", []))
    for om in online_methods:
        base_algo = om.split("/")[0]
        if base_algo not in registered and om not in registered:
            issues.append(f"Notice: Online method '{om}' might not match a registered agent ({registered}).")
            
    offline_methods = parse_method_list(cfg.get("offline_methods", []))
    for ofm in offline_methods:
        base_algo = ofm.split("/")[0]
        if base_algo not in registered and ofm not in registered:
            issues.append(f"Notice: Offline method '{ofm}' might not match a registered agent ({registered}).")
            
    # 3. Check offline dataset availability
    if mode_type == "offline" or offline_methods:
        offline_datasets = parse_method_list(cfg.get("offline_datasets", []))
        for ds in offline_datasets:
            # If dataset will be generated in online phase, that's fine
            if ds in online_methods or ds.replace("/", "_") in [m.replace("/", "_") for m in online_methods]:
                continue
            try:
                yaml_ds_path = cfg.mode.get("dataset_path", None) if hasattr(cfg, "mode") else None
                resolve_dataset_path(ds, group=cfg.get("group", ""), experiment_id=cfg.get("experiment_id", ""), yaml_ds_path=yaml_ds_path)
            except Exception as e:
                issues.append(f"Dataset check: {e}")
                
    # 4. Check Optuna direction if sweep
    if is_sweep:
        sweeper = cfg.get("hydra", {}).get("sweeper", {})
        direction = sweeper.get("direction", None)
        if is_offline_only and direction == "maximize":
            issues.append(
                f"Notice: Optuna direction is 'maximize' on offline-only experiment '{experiment_name}'. "
                f"Offline experiments monitor 'val/loss' and should use 'direction: minimize'."
            )
            
    if fatal_errors:
        raise ValueError("\n".join(["[Fatal Config Error] " + err for err in fatal_errors]))
        
    return issues
