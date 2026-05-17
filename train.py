import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize Agent
    agent_cfg = cfg.agent
    # Handle nesting from Hydra inheritance (e.g., agent.agent.name)
    def get_algo_name(acfg):
        if "name" in acfg:
            return acfg.name
        if "agent" in acfg:
            return get_algo_name(acfg.agent)
        return None
    
    base_algo_name = get_algo_name(agent_cfg)

    if base_algo_name == "ppo":
        from src.methods.ppo_agent import PPOAgent
        model = PPOAgent(cfg)
    elif base_algo_name == "blendrl":
        from src.methods.blendrl_agent import BlendRLAgent
        model = BlendRLAgent(cfg)
    elif base_algo_name == "iql":
        from src.methods.iql_agent import IQLAgent
        model = IQLAgent(cfg)
    elif base_algo_name == "blendrl_iql":
        from src.methods.blendrl_iql_agent import BlendRLIQLAgent
        model = BlendRLIQLAgent(cfg)
    else:
        raise ValueError(f"Unknown agent algorithm: {base_algo_name}")
    
    # Data Module
    from src.data.rl_data_module import RLDataModule
    datamodule = RLDataModule(cfg)
    
    # Loggers
    log_dir = os.path.join("results/logs", cfg.experiment_id)
    tb_dir = os.path.join("results/tensorboard", cfg.experiment_id)
    loggers = [
        CSVLogger(log_dir, name=cfg.agent.name),
        TensorBoardLogger(tb_dir, name=cfg.agent.name, default_hp_metric=False)
    ]
    
    # Callbacks
    from src.utils import EnvironmentEvaluatorCallback
    ckpt_dir = os.path.join("results/checkpoints", cfg.experiment_id, cfg.agent.name)
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

    trainer = L.Trainer(
        **cfg.get("trainer", {
            "max_epochs": max_epochs,
            "accelerator": "auto",
            "devices": 1,
            "limit_train_batches": 1.0,
            "limit_val_batches": limit_val_batches,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "log_every_n_steps": 1
        }),
        logger=loggers,
        callbacks=callbacks
    )
    
    trainer.fit(model, datamodule=datamodule)

    # Return metric for Optuna
    if "eval/reward" in trainer.callback_metrics:
        return trainer.callback_metrics["eval/reward"].item()
    return 0.0

if __name__ == "__main__":
    main()
