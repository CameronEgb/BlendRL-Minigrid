import os
import sys

# Ensure project root and src are in PYTHONPATH for Slurm/Cluster
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    agent_seed = cfg.agent.get("seed", cfg.seed)
    if isinstance(agent_seed, DictConfig): # Handle potential nesting
        agent_seed = agent_seed.get("seed", cfg.seed)
    L.seed_everything(agent_seed)
    
    # Initialize Agent
    agent_cfg = cfg.agent
    # Handle nesting from Hydra inheritance (e.g., agent.agent.name)
    def get_algo_name(acfg):
        # Recursively look for 'algorithm' then 'name'
        if isinstance(acfg, (dict, DictConfig)):
            if "algorithm" in acfg:
                return acfg.algorithm
            if "agent" in acfg:
                res = get_algo_name(acfg.agent)
                if res: return res
            if "name" in acfg:
                return acfg.name
        return None
    
    base_algo_name = get_algo_name(agent_cfg)
    print(f"Extracted algorithm name: {base_algo_name}")

    if not base_algo_name:
        raise ValueError("Could not extract algorithm name from config.")

    if base_algo_name.startswith("ppo"):
        from src.methods.ppo_agent import PPOAgent
        model = PPOAgent(cfg)
    elif base_algo_name.startswith("blendrl_iql"):
        from src.methods.blendrl_iql_agent import BlendRLIQLAgent
        model = BlendRLIQLAgent(cfg)
    elif base_algo_name.startswith("blendrl_cql"):
        from src.methods.blendrl_cql_agent import BlendRLCQLAgent
        model = BlendRLCQLAgent(cfg)
    elif base_algo_name.startswith("cql"):
        from src.methods.cql_agent import CQLAgent
        model = CQLAgent(cfg)
    elif base_algo_name.startswith("blendrl"):
        from src.methods.blendrl_agent import BlendRLAgent
        model = BlendRLAgent(cfg)
    elif base_algo_name.startswith("iql"):
        from src.methods.iql_agent import IQLAgent
        model = IQLAgent(cfg)
    elif base_algo_name.startswith("cew"):
        from src.methods.cew_agent import CEWAgent
        model = CEWAgent(cfg)
    else:
        raise ValueError(f"Unknown agent algorithm: {base_algo_name}")
    
    # Print Architecture Information
    print("\n" + "="*30)
    print("      AGENT ARCHITECTURE")
    print("="*30)
    print(model)
    
    if hasattr(model, "model") and hasattr(model.model, "_print"):
        print("\n" + "="*30)
        print("      LOGIC RULES & BLENDER")
        print("="*30)
        model.model._print()
    print("="*30 + "\n")
    
    # Data Module
    from src.data.rl_data_module import RLDataModule
    datamodule = RLDataModule(cfg)
    
    # Loggers
    log_dir = os.path.join("results/logs", cfg.group, cfg.experiment_id)
    tb_dir = os.path.join("results/tensorboard", cfg.group, cfg.experiment_id)
    
    loggers = [CSVLogger(log_dir, name=cfg.agent.name)]
    try:
        tb_logger = TensorBoardLogger(tb_dir, name=cfg.agent.name, default_hp_metric=False)
        # Access self.experiment to force the import of tensorboard and catch failures early
        _ = tb_logger.experiment
        loggers.append(tb_logger)
    except Exception as e:
        print(f"\n[Warning] TensorBoardLogger failed to load due to environment dependency issues: {e}")
        print("[Warning] Falling back to CSVLogger only. (Your plots will still work as they read from CSV logs).\n")
    
    # Callbacks
    from src.utils import EnvironmentEvaluatorCallback
    # Use trial-specific subdirectory for checkpoints to avoid auto-recovery collisions
    trial_id = cfg.get("hydra", {}).get("job", {}).get("num", "0")
    ckpt_dir = os.path.join("results/checkpoints", cfg.group, cfg.experiment_id, cfg.agent.name, str(trial_id))
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best_model",
            monitor="eval/reward",
            mode="max",
            save_top_k=1
        ),
        EnvironmentEvaluatorCallback(cfg)
    ]
    
    # Trainer
    if cfg.mode.type == "online":
        # Check if the model has its own num_envs and num_steps (agent overrides)
        num_envs = getattr(model, "num_envs", cfg.env.num_envs)
        num_steps = getattr(model, "num_steps", cfg.env.num_steps)
        batch_size = num_envs * num_steps
        
        import math
        max_epochs = math.ceil(cfg.total_timesteps / batch_size)
        if max_epochs == 0: max_epochs = 1
        
        limit_val_batches = 0
        check_val_every_n_epoch = 1000000 
    else:
        epochs_per_interval = cfg.agent.get("epochs_per_interval", 1)
        max_epochs = cfg.intervals_count * epochs_per_interval
        limit_val_batches = 1
        check_val_every_n_epoch = 1  # Check every epoch, let callback handle frequency

    # Trainer Configuration
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": 1,
        "limit_train_batches": 1.0,
        "limit_val_batches": limit_val_batches,
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "log_every_n_steps": 1
    }
    
    # Merge with overrides from the config file if they exist
    if "trainer" in cfg and cfg.trainer is not None:
        trainer_overrides = OmegaConf.to_container(cfg.trainer, resolve=True)
        trainer_kwargs.update(trainer_overrides)

    trainer = L.Trainer(
        **trainer_kwargs,
        logger=loggers,
        callbacks=callbacks
    )
    
    # Determine checkpoint for recovery
    ckpt_path = None
    if cfg.get("recover", False):
        potential_ckpt = os.path.join(ckpt_dir, "best_model.ckpt")
        if os.path.exists(potential_ckpt):
            print(f"Recovering from checkpoint: {potential_ckpt}")
            ckpt_path = potential_ckpt
        else:
            # Fallback to the non-trial-specific directory if trial-specific doesn't exist yet
            legacy_ckpt = os.path.join("results/checkpoints", cfg.group, cfg.experiment_id, cfg.agent.name, "best_model.ckpt")
            if os.path.exists(legacy_ckpt):
                 print(f"Recovering from legacy checkpoint: {legacy_ckpt}")
                 ckpt_path = legacy_ckpt

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Return metric for Optuna
    if "eval/reward" in trainer.callback_metrics:
        return trainer.callback_metrics["eval/reward"].item()
    return 0.0

if __name__ == "__main__":
    main()
