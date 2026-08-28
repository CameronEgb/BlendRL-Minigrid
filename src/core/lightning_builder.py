import os
import sys
import time
import json
import torch
import signal

from omegaconf import OmegaConf, DictConfig
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

class SaveInitialCheckpointCallback(Callback):
    """Callback to save an initial, untrained model checkpoint before training starts."""
    def __init__(self, ckpt_dir, cfg):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg

    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        init_ckpt = os.path.join(self.ckpt_dir, "best_model.ckpt")
        trainer.save_checkpoint(init_ckpt)
        from hydra.utils import get_original_cwd
        try:
            base_root = get_original_cwd()
        except Exception:
            base_root = os.getcwd()
        parent_ckpt_root = os.path.join(base_root, "results/checkpoints", self.cfg.group, self.cfg.experiment_id)
        os.makedirs(parent_ckpt_root, exist_ok=True)
        named_ckpt = os.path.join(parent_ckpt_root, f"{self.cfg.agent.name}.ckpt")
        import shutil
        shutil.copy2(init_ckpt, named_ckpt)
        print(f"[Init Checkpoint] Saved initial model checkpoint to: {init_ckpt}")

def print_hardware_diagnostics():
    """Print GPU hardware info and diagnostics before training."""
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

def infer_dynamic_timesteps(cfg):
    """Dynamically infer total timesteps based on offline dataset size."""
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

def setup_loggers(cfg, base_root):
    log_dir = os.path.join(base_root, "results/logs", cfg.group, cfg.experiment_id)
    tb_dir = os.path.join(base_root, "results/tensorboard", cfg.group, cfg.experiment_id)
    loggers = [CSVLogger(log_dir, name=cfg.agent.name)]
    tb_logger = TensorBoardLogger(tb_dir, name=cfg.agent.name, default_hp_metric=False)
    _ = tb_logger.experiment
    loggers.append(tb_logger)
    return loggers

def build_trainer(cfg, model=None):
    print_hardware_diagnostics()
    infer_dynamic_timesteps(cfg)

    agent_seed = cfg.agent.get("seed", cfg.seed)
    if isinstance(agent_seed, DictConfig):
        agent_seed = agent_seed.get("seed", cfg.seed)
    L.seed_everything(agent_seed)

    from hydra.utils import get_original_cwd
    try:
        base_root = get_original_cwd()
    except Exception:
        base_root = os.getcwd()

    loggers = setup_loggers(cfg, base_root)
    
    from hydra.core.hydra_config import HydraConfig
    trial_id = "0"
    if HydraConfig.initialized():
        try:
            trial_id = str(HydraConfig.get().job.num)
        except Exception:
            pass
    ckpt_dir = os.path.join(base_root, "results/checkpoints", cfg.group, cfg.experiment_id, cfg.agent.name, trial_id)

    def graceful_shutdown(signum, frame):
        print(f"\n[SIGTERM] Received termination signal {signum}. Attempting graceful shutdown...")
        # Since trainer is constructed here, we need a global reference or just sys.exit(0)
        # We will handle it by just capturing SIGTERM for basic shutdown
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, graceful_shutdown)
        signal.signal(signal.SIGUSR1, graceful_shutdown)
    except Exception:
        pass 

    is_offline_only = cfg.env.get("offline_only", False)

    callbacks = [SaveInitialCheckpointCallback(ckpt_dir, cfg)]
    if is_offline_only:
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
        from src.core.callbacks import EnvironmentEvaluatorCallback
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

    if cfg.mode.type == "online":
        num_envs = getattr(model, "num_envs", cfg.env.num_envs) if model else cfg.env.num_envs
        num_steps = getattr(model, "num_steps", cfg.env.num_steps) if model else getattr(cfg.env, "num_steps", 256)
        batch_size = num_envs * num_steps
        
        import math
        max_epochs = math.ceil(cfg.total_timesteps / batch_size) if batch_size > 0 else 1
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
    
    if "trainer" in cfg and cfg.trainer is not None:
        trainer_overrides = OmegaConf.to_container(cfg.trainer, resolve=True)
        trainer_kwargs.update(trainer_overrides)

    trainer = L.Trainer(
        **trainer_kwargs,
        logger=loggers,
        callbacks=callbacks
    )

    ckpt_path = None
    if cfg.get("recover", False):
        potential_ckpt = os.path.join(ckpt_dir, "best_model.ckpt")
        if os.path.exists(potential_ckpt):
            print(f"Recovering from checkpoint: {potential_ckpt}")
            ckpt_path = potential_ckpt

    return trainer, ckpt_dir, ckpt_path

def finalize_training(trainer, cfg, ckpt_dir, training_time, start_time, end_time):
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
    
    from src.core.metadata import collect_run_metadata, save_git_diff
    import datetime
    from hydra.core.hydra_config import HydraConfig
    
    hydra_output_dir = None
    if HydraConfig.initialized():
        try:
            hydra_output_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            pass

    start_time_iso = datetime.datetime.fromtimestamp(start_time, datetime.timezone.utc).isoformat() if start_time else None
    end_time_iso = datetime.datetime.fromtimestamp(end_time, datetime.timezone.utc).isoformat() if end_time else None

    meta = collect_run_metadata(cfg)
    save_git_diff(ckpt_dir)

    runtime_info = {
        "agent": str(cfg.agent.name),
        "experiment_id": str(cfg.experiment_id),
        "group": str(cfg.group),
        "training_time_seconds": round(training_time, 2),
        "start_time": start_time,
        "end_time": end_time,
        "start_time_iso": start_time_iso,
        "end_time_iso": end_time_iso,
        "hydra_output_dir": hydra_output_dir,
        **meta,
        **gpu_stats
    }

    def atomic_json_dump(obj, path):
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def atomic_yaml_save(cfg_obj, path):
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            OmegaConf.save(config=cfg_obj, f=f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    atomic_json_dump(runtime_info, os.path.join(ckpt_dir, "runtime.json"))

    try:
        atomic_yaml_save(cfg, os.path.join(ckpt_dir, "config.yaml"))
        exp_ckpt_root = os.path.join("results/checkpoints", cfg.group, cfg.experiment_id)
        os.makedirs(exp_ckpt_root, exist_ok=True)
        atomic_yaml_save(cfg, os.path.join(exp_ckpt_root, "config.yaml"))
        
        if hydra_output_dir:
            import shutil
            overrides_path = os.path.join(hydra_output_dir, ".hydra", "overrides.yaml")
            if os.path.exists(overrides_path):
                shutil.copy2(overrides_path, os.path.join(ckpt_dir, "overrides.yaml"))
    except Exception as e:
        print(f"Notice: Could not save checkpoint config.yaml or overrides.yaml: {e}")

    for lg in active_loggers:
        if hasattr(lg, "log_dir") and lg.log_dir:
            os.makedirs(lg.log_dir, exist_ok=True)
            atomic_json_dump(runtime_info, os.path.join(lg.log_dir, "runtime.json"))
            try:
                atomic_yaml_save(cfg, os.path.join(lg.log_dir, "config.yaml"))
                if hydra_output_dir:
                    overrides_path = os.path.join(hydra_output_dir, ".hydra", "overrides.yaml")
                    if os.path.exists(overrides_path):
                        import shutil
                        shutil.copy2(overrides_path, os.path.join(lg.log_dir, "overrides.yaml"))
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
        atomic_yaml_save(cfg, os.path.join(exp_log_root, "config.yaml"))
    except Exception as e:
        print(f"Notice: Could not save logger config.yaml: {e}")

    final_ckpt_target = os.path.join(ckpt_dir, "best_model.ckpt")
    if not os.path.exists(final_ckpt_target):
        print(f"Saving final trained model checkpoint to: {final_ckpt_target}")
        trainer.save_checkpoint(final_ckpt_target)

    is_offline_only = cfg.env.get("offline_only", False)
    metric_name = cfg.env.get("monitor_metric", "eval/reward")
    if metric_name in trainer.callback_metrics:
        return trainer.callback_metrics[metric_name].item()
    return 0.0

