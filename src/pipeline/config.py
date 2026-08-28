"""Configuration utilities for the pipeline.

Provides functions for parsing method lists and normalizing agent names.
"""


def normalize_agent_name(agent_config: str) -> str:
    """Convert hierarchical agent config paths and dataset IDs to filesystem-safe and Hydra-safe names.
    e.g. 'blendrl_cql/human_cew' -> 'blendrl_cql_human_cew'
         'ex132(w)' -> 'ex132_w'
    This must match agent.name as set in the Hydra overrides."""
    return (
        agent_config.replace("/", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace("__", "_")
        .rstrip("_")
    )


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
    if not matches:
        raise ValueError(f"Experiment config '{clean_input}.yaml' not found in {exp_dir}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous experiment name '{clean_input}'. Found multiple matches: {[str(m) for m in matches]}. Please specify the exact group/experiment path.")
        
    rel = matches[0].relative_to(exp_dir)
    return str(rel.with_suffix(""))



