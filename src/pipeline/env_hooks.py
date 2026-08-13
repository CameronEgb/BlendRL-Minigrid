"""Environment-specific hooks for the pipeline.

Each environment can optionally define a hooks module at in/envs/<env_name>/hooks.py
containing functions that customize pipeline behavior. This module provides the
loader and default no-op implementations.
"""
import importlib.util
import os


class DefaultHooks:
    """No-op hooks for environments that don't need customization."""
    
    @staticmethod
    def transform_rewards(reader, cfg):
        """Called after DatasetReader loads data. Can modify reader.rewards in-place."""
        pass
    
    @staticmethod  
    def preprocess_dataset(cfg):
        """Called before training. Environment-specific dataset preprocessing."""
        pass
    
    @staticmethod
    def post_training_eval(cfg, local_val):
        """Called after training completes. Environment-specific evaluation."""
        pass


def load_env_hooks(env_name: str):
    """Load environment-specific hooks, falling back to DefaultHooks.
    
    Looks for a hooks module at in/envs/<env_name>/hooks.py.
    If found, returns an instance of the Hooks class from that module.
    Otherwise returns DefaultHooks.
    """
    hooks_path = os.path.join("in", "envs", env_name, "hooks.py")
    if os.path.exists(hooks_path):
        try:
            spec = importlib.util.spec_from_file_location(f"in.envs.{env_name}.hooks", hooks_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'Hooks'):
                return module.Hooks()
        except Exception as e:
            print(f"Warning: Failed to load hooks for {env_name}: {e}")
    return DefaultHooks()
