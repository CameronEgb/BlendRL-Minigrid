"""Configuration utilities for the pipeline.

Provides functions for parsing method lists and normalizing agent names.
"""


def normalize_agent_name(agent_config: str) -> str:
    """Convert hierarchical agent config paths to filesystem-safe names.
    e.g. 'blendrl_cql/human_cew' -> 'blendrl_cql_human_cew'
    This must match agent.name as set in the Hydra overrides."""
    return agent_config.replace("/", "_")


def parse_method_list(val):
    """Parse a method list from Hydra config.
    Hydra/YAML returns a Python list for `[a, b]` syntax but a string for `"a, b"` syntax.
    This function handles both forms."""
    if not val: return []
    if isinstance(val, (list, tuple)): return list(val)
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return list(val)
    return [item.strip() for item in str(val).split(",") if item.strip()]


def resolve_experiment_config_name(exp_input: str) -> str:
    """Resolve an experiment name (e.g. 'mimic_cql' or 'mimic/mimic_cql')
    to its relative Hydra config path inside in/config/experiment/."""
    from pathlib import Path
    exp_dir = Path("in/config/experiment")
    if not exp_dir.exists():
        return exp_input
        
    clean_input = exp_input[:-5] if exp_input.endswith(".yaml") else exp_input
    direct_path = exp_dir / f"{clean_input}.yaml"
    if direct_path.exists():
        return clean_input
        
    # Search recursively in group subdirectories
    matches = list(exp_dir.glob(f"**/{clean_input}.yaml"))
    if matches:
        rel = matches[0].relative_to(exp_dir)
        return str(rel.with_suffix(""))
        
    return clean_input


def filter_pipeline_args(extra_args, experiment: str, exp_id: str = None):
    """Sanitize extra CLI arguments and build Hydra compose overrides.
    
    Returns:
        tuple[list[str], list[str]]: (sanitized_extra_args, overrides_for_compose)
    """
    sanitized_extra_args = []
    overrides_for_compose = [f"+experiment={experiment}"]

    if exp_id:
        sanitized_extra_args.append(f"++experiment_id={exp_id}")
        overrides_for_compose.append(f"++experiment_id={exp_id}")
    
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
            is_hydra = sanitized_arg.startswith("hydra.") or "/" in sanitized_arg.split("=")[0]
            if not (is_sweep or is_hydra):
                overrides_for_compose.append(sanitized_arg)
        else:
            # Flags like --multirun or -m should be passed to subprocess but NOT to compose
            sanitized_extra_args.append(arg)

    return sanitized_extra_args, overrides_for_compose


def get_python_executable() -> str:
    """Returns the path to the python executable to use for all subprocesses."""
    import os
    import sys
    project_root = os.getcwd()
    venv_python = os.path.join(project_root, "venv", "bin", "python3")
    
    # Check if we should use python3.13 specifically if it exists
    venv_python_13 = os.path.join(project_root, "venv", "bin", "python3.13")
    if os.path.exists(venv_python_13):
        return venv_python_13
        
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable
