import importlib
import os
import pkgutil

TASK_REGISTRY = {}

def register_task(*prefixes):
    """Decorator to register a task function under one or more prefixes.
    
    Usage:
        @register_task("reciprocal_refinement")
        def run_reciprocal_task(cfg, args, context):
            ...
    """
    def decorator(fn):
        for prefix in prefixes:
            TASK_REGISTRY[prefix] = fn
        return fn
    return decorator

def get_task(task_name: str):
    """Resolve a task name to its registered function using longest-prefix matching.
    
    Args:
        task_name: Task identifier (e.g., "early_prediction_sweep").
    
    Returns:
        The registered task function.
    """
    if not TASK_REGISTRY:
        auto_discover_tasks()

    if task_name in TASK_REGISTRY:
        return TASK_REGISTRY[task_name]
    
    # Longest-prefix match
    matches = [
        (prefix, fn) for prefix, fn in TASK_REGISTRY.items()
        if task_name == prefix or task_name.startswith(prefix + "_")
    ]
    
    if not matches:
        registered = sorted(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task: '{task_name}'. "
            f"Registered prefixes: {registered}"
        )
    
    return max(matches, key=lambda x: len(x[0]))[1]

def list_tasks():
    """Return sorted list of registered task prefixes."""
    auto_discover_tasks()
    return sorted(TASK_REGISTRY.keys())

def auto_discover_tasks():
    """Import all modules in src/pipeline/ to trigger @register_task decorators."""
    import sys
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(pipeline_dir)
    project_root = os.path.dirname(src_dir)
    for p in (project_root, src_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
            
    for module_info in pkgutil.iter_modules([pipeline_dir]):
        if module_info.name.startswith("_") or module_info.name == "task_registry":
            continue
        try:
            importlib.import_module(f"src.pipeline.{module_info.name}")
        except Exception:
            try:
                importlib.import_module(f"pipeline.{module_info.name}")
            except Exception as e:
                import logging; logging.getLogger(__name__).debug(f"Could not import {module_info.name}: {e}")
