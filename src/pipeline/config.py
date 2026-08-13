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
