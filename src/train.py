import os
import sys

# Ensure project root and src are in PYTHONPATH for Slurm/Cluster
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import omegaconf
try:
    torch.serialization.add_safe_globals([
        omegaconf.dictconfig.DictConfig,
        omegaconf.listconfig.ListConfig,
        omegaconf.base.Container,
        omegaconf.nodes.UntypedNode,
    ])
except Exception:
    pass
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os

@hydra.main(version_base=None, config_path="../in/config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Propagate reward type to environment variable
    if "env" in cfg and "reward_type" in cfg.env:
        os.environ["MIMIC_REWARD_TYPE"] = str(cfg.env.reward_type)
    if "env" in cfg and "name" in cfg.env:
        os.environ["BLENDRL_ENV_NAME"] = str(cfg.env.name)

    # Dynamic total_timesteps inference for offline mode
    if cfg.mode.type == "offline":
        ds_path = cfg.mode.get("dataset_path", None)
        if ds_path and os.path.exists(ds_path):
            try:
                from src.dataset_utils import DatasetReader
                temp_reader = DatasetReader(ds_path)
                ds_len = len(temp_reader)
                if ds_len > 0:
                    current_tt = cfg.get("total_timesteps", None)
                    if current_tt is None or str(current_tt).lower() in ("auto", "-1", "none") or current_tt != ds_len:
                        print(f"\n[Dynamic Config] Detected offline dataset with {ds_len} transitions at '{ds_path}'.")
                        print(f"[Dynamic Config] Setting total_timesteps = {ds_len}.\n")
                        OmegaConf.set_struct(cfg, False)
                        cfg.total_timesteps = ds_len
                        OmegaConf.set_struct(cfg, True)
            except Exception as e:
                print(f"[Dynamic Config Warning] Failed to dynamically inspect dataset size: {e}")
    
    # Set seed
    agent_seed = cfg.agent.get("seed", cfg.seed)
    if isinstance(agent_seed, DictConfig): # Handle potential nesting
        agent_seed = agent_seed.get("seed", cfg.seed)
    L.seed_everything(agent_seed)
    
    # Initialize Agent via Registry
    # Auto-discover all agent classes (triggers @register_agent decorators)
    from src.methods.registry import auto_discover, get_agent_class
    auto_discover()
    
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

    AgentClass = get_agent_class(base_algo_name)
    print(f"Resolved agent class: {AgentClass.__name__}")
    model = AgentClass(cfg)
    
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
    from hydra.core.hydra_config import HydraConfig
    trial_id = "0"
    if HydraConfig.initialized():
        try:
            trial_id = str(HydraConfig.get().job.num)
        except Exception:
            pass
    ckpt_dir = os.path.join("results/checkpoints", cfg.group, cfg.experiment_id, cfg.agent.name, trial_id)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best_model",
            monitor="eval/reward",
            mode="max",
            save_top_k=1,
            enable_version_counter=False
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

    import time
    import json

    start_time = time.time()
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n[Training Complete] Total execution time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    try:
        if trainer.logger is not None:
            trainer.logger.log_metrics({"training_time_seconds": training_time}, step=trainer.global_step)
    except Exception:
        pass

    os.makedirs(ckpt_dir, exist_ok=True)
    runtime_info = {
        "agent": str(cfg.agent.name),
        "experiment_id": str(cfg.experiment_id),
        "group": str(cfg.group),
        "training_time_seconds": round(training_time, 2),
        "start_time": start_time,
        "end_time": end_time
    }
    with open(os.path.join(ckpt_dir, "runtime.json"), "w") as f:
        json.dump(runtime_info, f, indent=2)

    if hasattr(trainer, "logger") and hasattr(trainer.logger, "log_dir") and trainer.logger.log_dir:
        os.makedirs(trainer.logger.log_dir, exist_ok=True)
        with open(os.path.join(trainer.logger.log_dir, "runtime.json"), "w") as f:
            json.dump(runtime_info, f, indent=2)

    # Guarantee final model checkpoint is saved to disk
    final_ckpt_target = os.path.join(ckpt_dir, "best_model.ckpt")
    if not os.path.exists(final_ckpt_target):
        print(f"Saving final trained model checkpoint to: {final_ckpt_target}")
        trainer.save_checkpoint(final_ckpt_target)

    # Return metric for Optuna
    if "eval/reward" in trainer.callback_metrics:
        return trainer.callback_metrics["eval/reward"].item()
    return 0.0

if __name__ == "__main__":
    main()
