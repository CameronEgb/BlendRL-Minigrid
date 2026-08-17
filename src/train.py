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
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

@hydra.main(version_base=None, config_path="../in/config", config_name="config")
def main(cfg: DictConfig):
    # Auto-infer experiment_id from Hydra task override if not explicitly specified
    if cfg.get("experiment_id", "default_exp") == "default_exp":
        try:
            from hydra.core.hydra_config import HydraConfig
            if HydraConfig.initialized():
                for override in HydraConfig.get().overrides.task:
                    if override.startswith("+experiment=") or override.startswith("experiment="):
                        exp_stem = Path(override.split("=")[-1]).stem
                        cfg.experiment_id = exp_stem
                        break
        except Exception:
            pass

    print(OmegaConf.to_yaml(cfg))
    
    # Propagate reward type to environment variable
    if "env" in cfg and "reward_type" in cfg.env:
        os.environ["MIMIC_REWARD_TYPE"] = str(cfg.env.reward_type)
    if "env" in cfg and "name" in cfg.env:
        os.environ["BLENDRL_ENV_NAME"] = str(cfg.env.name)

    # Hardware & GPU Diagnostics Setup
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        device_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_idx)
        total_vram_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
        print("\n" + "="*50)
        print("      GPU HARDWARE DIAGNOSTICS")
        print("="*50)
        print(f"  Device:          {gpu_name} (ID: {device_idx})")
        print(f"  Total VRAM:      {total_vram_gb:.2f} GB")
        print(f"  CUDA Version:    {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        print("="*50 + "\n")

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
    
    # Callbacks & Evaluation Architecture
    from hydra.core.hydra_config import HydraConfig
    trial_id = "0"
    if HydraConfig.initialized():
        try:
            trial_id = str(HydraConfig.get().job.num)
        except Exception:
            pass
    ckpt_dir = os.path.join("results/checkpoints", cfg.group, cfg.experiment_id, cfg.agent.name, trial_id)
    
    is_offline_only = cfg.env.get("offline_only", False) or cfg.env.name in ["mimic", "pyrenees"]
    
    callbacks = []
    if is_offline_only:
        print(f"\n[Environment Setup] Detected offline-only environment '{cfg.env.name}'.")
        print("[Environment Setup] Bypassing simulated gym rollouts; monitoring validation loss for checkpointing.\n")
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best_model",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                enable_version_counter=False
            )
        )
    else:
        from src.utils import EnvironmentEvaluatorCallback
        callbacks.extend([
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best_model",
                monitor="eval/reward",
                mode="max",
                save_top_k=1,
                enable_version_counter=False
            ),
            EnvironmentEvaluatorCallback(cfg)
        ])
    
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
        intervals_count = 1 if is_offline_only else cfg.get("intervals_count", 1)
        max_epochs = intervals_count * epochs_per_interval
        eval_interval_epochs = cfg.agent.get("eval_interval_epochs", 1)
        limit_val_batches = 1.0
        check_val_every_n_epoch = eval_interval_epochs if eval_interval_epochs else 1

    # Trainer Configuration
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": 1,
        "limit_train_batches": 1.0,
        "limit_val_batches": limit_val_batches,
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 0,
        "enable_progress_bar": False if not sys.stdout.isatty() else True,
    }
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        precision_setting = cfg.get("precision", "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed")
        trainer_kwargs["precision"] = precision_setting
    
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

    gpu_stats = {}
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        total_vram_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
        peak_alloc_gb = torch.cuda.max_memory_allocated(device_idx) / (1024**3)
        peak_res_gb = torch.cuda.max_memory_reserved(device_idx) / (1024**3)
        mem_eff_pct = (peak_alloc_gb / total_vram_gb) * 100.0 if total_vram_gb > 0 else 0.0

        gpu_stats = {
            "gpu_device": torch.cuda.get_device_name(device_idx),
            "gpu_total_vram_gb": round(total_vram_gb, 2),
            "gpu_peak_alloc_gb": round(peak_alloc_gb, 3),
            "gpu_peak_reserved_gb": round(peak_res_gb, 3),
            "gpu_mem_efficiency_pct": round(mem_eff_pct, 2)
        }

        print("\n" + "="*50)
        print("      GPU MEMORY & RESOURCE FOOTPRINT")
        print("="*50)
        print(f"  GPU Device:        {gpu_stats['gpu_device']}")
        print(f"  Total VRAM:        {gpu_stats['gpu_total_vram_gb']:.2f} GB")
        print(f"  Peak Allocated:    {gpu_stats['gpu_peak_alloc_gb']:.3f} GB")
        print(f"  Peak Reserved:     {gpu_stats['gpu_peak_reserved_gb']:.3f} GB")
        print(f"  VRAM Footprint %:  {gpu_stats['gpu_mem_efficiency_pct']:.2f}%")
        print("="*50 + "\n")

    active_loggers = trainer.loggers if (hasattr(trainer, "loggers") and trainer.loggers) else ([trainer.logger] if trainer.logger else [])
    for lg in active_loggers:
        try:
            if hasattr(lg, "log_metrics"):
                metrics_to_log = {"training_time_seconds": training_time}
                if gpu_stats:
                    metrics_to_log.update({
                        "gpu/peak_alloc_gb": gpu_stats["gpu_peak_alloc_gb"],
                        "gpu/peak_reserved_gb": gpu_stats["gpu_peak_reserved_gb"],
                    })
                lg.log_metrics(metrics_to_log, step=trainer.global_step)
        except Exception:
            pass

    os.makedirs(ckpt_dir, exist_ok=True)
    runtime_info = {
        "agent": str(cfg.agent.name),
        "experiment_id": str(cfg.experiment_id),
        "group": str(cfg.group),
        "training_time_seconds": round(training_time, 2),
        "start_time": start_time,
        "end_time": end_time,
        **gpu_stats
    }
    with open(os.path.join(ckpt_dir, "runtime.json"), "w") as f:
        json.dump(runtime_info, f, indent=2)

    try:
        OmegaConf.save(config=cfg, f=os.path.join(ckpt_dir, "config.yaml"))
        exp_ckpt_root = os.path.join("results/checkpoints", cfg.group, cfg.experiment_id)
        os.makedirs(exp_ckpt_root, exist_ok=True)
        OmegaConf.save(config=cfg, f=os.path.join(exp_ckpt_root, "config.yaml"))
    except Exception as e:
        print(f"Notice: Could not save checkpoint config.yaml: {e}")

    for lg in active_loggers:
        if hasattr(lg, "log_dir") and lg.log_dir:
            os.makedirs(lg.log_dir, exist_ok=True)
            with open(os.path.join(lg.log_dir, "runtime.json"), "w") as f:
                json.dump(runtime_info, f, indent=2)
            try:
                OmegaConf.save(config=cfg, f=os.path.join(lg.log_dir, "config.yaml"))
            except Exception:
                pass
        if hasattr(lg, "save"):
            try:
                lg.save()
            except Exception:
                pass

    exp_log_root = os.path.join("results/logs", cfg.group, cfg.experiment_id)
    os.makedirs(exp_log_root, exist_ok=True)
    try:
        OmegaConf.save(config=cfg, f=os.path.join(exp_log_root, "config.yaml"))
    except Exception as e:
        print(f"Notice: Could not save logger config.yaml: {e}")

    # Guarantee final model checkpoint is saved to disk
    final_ckpt_target = os.path.join(ckpt_dir, "best_model.ckpt")
    if not os.path.exists(final_ckpt_target):
        print(f"Saving final trained model checkpoint to: {final_ckpt_target}")
        trainer.save_checkpoint(final_ckpt_target)

    # Return metric for Optuna
    if is_offline_only:
        if "val/robust_loss" in trainer.callback_metrics:
            return trainer.callback_metrics["val/robust_loss"].item()
        if "val/loss" in trainer.callback_metrics:
            return trainer.callback_metrics["val/loss"].item()
        if "losses/bellman_loss" in trainer.callback_metrics:
            return trainer.callback_metrics["losses/bellman_loss"].item()
        if "losses/total_loss" in trainer.callback_metrics:
            return trainer.callback_metrics["losses/total_loss"].item()
    else:
        if "eval/reward" in trainer.callback_metrics:
            return trainer.callback_metrics["eval/reward"].item()
    return 0.0

if __name__ == "__main__":
    main()
