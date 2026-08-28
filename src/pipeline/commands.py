"""Single source of truth for training command construction.

Eliminates 3x duplication of online/offline Hydra override lists
across local_runner.py, slurm_runner.py (consolidated), and slurm_runner.py (standalone).
"""


def build_online_overrides(experiment, agent_config, agent_name, dataset_path,
                           local_val, study_name=None, extra_args=None,
                           cfg=None):
    """Build Hydra override list for online training.

    Args:
        experiment: Experiment config name (e.g. 'cartpole/cp_test')
        agent_config: Agent Hydra config path (e.g. 'ppo/dnn')
        agent_name: Normalized agent name for filesystem (e.g. 'ppo_dnn')
        dataset_path: Where to save the dataset
        local_val: Whether running locally (bool)
        study_name: Optuna study name (optional)
        extra_args: Additional Hydra overrides from CLI
        cfg: Full Hydra config (passed to hooks)

    Returns:
        list[str]: Hydra override arguments
    """
    overrides = [
        f"+experiment={experiment}",
        "mode=online",
        f"agent={agent_config}",
        f"++agent.name={agent_name}",
        f"++dataset_path={dataset_path}",
    ]
    if study_name:
        overrides.append(f"++hydra.sweeper.study_name={study_name}")

    if extra_args:
        overrides.extend(extra_args)
    return overrides


def build_offline_overrides(experiment, agent_config, agent_name, dataset_path,
                            local_val=True, study_name=None, extra_args=None,
                            cfg=None, dataset_id=None):
    """Build Hydra override list for offline training.

    Args:
        experiment: Experiment config name
        agent_config: Agent Hydra config path
        agent_name: Normalized agent name for filesystem
        dataset_path: Path to offline dataset
        local_val: Unused (deprecated)
        study_name: Optuna study name (optional)
        extra_args: Additional Hydra overrides from CLI
        cfg: Full Hydra config (passed to hooks)
        dataset_id: Raw dataset identifier (passed to hooks for env-specific overrides)

    Returns:
        list[str]: Hydra override arguments
    """
    overrides = [
        f"+experiment={experiment}",
        "mode=offline",
        f"agent={agent_config}",
        f"++agent.name={agent_name}",
        f"++mode.dataset_path={dataset_path}",
    ]
    if study_name:
        overrides.append(f"++hydra.sweeper.study_name={study_name}")

    if extra_args:
        overrides.extend(extra_args)
    return overrides


def get_sweep_direction(cfg, mode: str) -> str:
    """Single source of truth for Optuna sweep direction.

    Checks cfg.hydra.sweeper.direction first (explicit override).
    Defaults: online -> 'maximize' (reward), offline -> 'minimize' (loss).
    """
    if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper"):
        explicit = cfg.hydra.sweeper.get("direction", None)
        if explicit:
            return str(explicit)
    # Default: online maximizes reward, offline minimizes loss
    return "maximize" if mode == "online" else "minimize"
