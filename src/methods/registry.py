"""
Agent Registry — Single source of truth for agent class lookup.

Agents self-register using the @register_agent decorator:
    @register_agent("iql")
    class IQLAgent(OfflineAgentBase):
        ...

The pipeline resolves agents via get_agent_class():
    AgentClass = get_agent_class("blendrl_iql_cp_tuned")
    # → Returns BlendRLIQLAgent (longest prefix match: "blendrl_iql")
"""

import importlib
import os
import pkgutil

AGENT_REGISTRY = {}


def register_agent(*prefixes):
    """Decorator to register an agent class under one or more algorithm prefixes.
    
    Usage:
        @register_agent("iql")
        class IQLAgent(OfflineAgentBase):
            ...
        
        @register_agent("blendrl_iql")
        class BlendRLIQLAgent(IQLAgent):
            ...
    """
    def decorator(cls):
        for prefix in prefixes:
            AGENT_REGISTRY[prefix] = cls
        return cls
    return decorator


def get_agent_class(algo_name: str):
    """Resolve an algorithm name to its registered agent class using longest-prefix matching.
    
    Args:
        algo_name: Algorithm identifier (e.g., "iql_cp_tuned", "blendrl_cql_human_cew").
    
    Returns:
        The registered agent class.
    
    Raises:
        ValueError: If no registered prefix matches the algorithm name.
    """
    if not AGENT_REGISTRY:
        auto_discover()

    # Try exact match first
    if algo_name in AGENT_REGISTRY:
        return AGENT_REGISTRY[algo_name]
    
    # Longest-prefix match: sort by key length descending, check startswith
    matches = [
        (prefix, cls) for prefix, cls in AGENT_REGISTRY.items()
        if algo_name == prefix or algo_name.startswith(prefix + "_")
    ]
    
    if not matches:
        registered = sorted(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent algorithm: '{algo_name}'. "
            f"Registered prefixes: {registered}"
        )
    
    # Return the class with the longest matching prefix
    return max(matches, key=lambda x: len(x[0]))[1]


def auto_discover():
    """Import all modules in src/methods/ to trigger @register_agent decorators.
    
    Called once at startup (e.g., in train.py) to ensure all agents are registered
    before get_agent_class() is used.
    """
    import sys
    methods_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(methods_dir)
    project_root = os.path.dirname(src_dir)
    for p in (project_root, src_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
            
    for module_info in pkgutil.iter_modules([methods_dir]):
        if module_info.name.startswith("_") or module_info.name in ("registry", "base_agent"):
            continue
        try:
            importlib.import_module(f"src.methods.{module_info.name}")
        except Exception as e:
            # Fall back to direct name import if package relative
            try:
                importlib.import_module(f"methods.{module_info.name}")
            except Exception:
                print(f"[Warning] Failed to auto-discover agent module '{module_info.name}': {e}")


def list_registered_agents():
    """Return sorted list of registered algorithm prefixes."""
    auto_discover()
    return sorted(AGENT_REGISTRY.keys())
