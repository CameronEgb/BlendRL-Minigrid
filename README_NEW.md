# Project Reorganization: New Structure

The project has been refactored to use **PyTorch Lightning** and **Hydra**. The previous manual orchestration scripts and duplicate training loops have been moved to `.legacy/`.

## New Project Structure

- `train.py`: The single entry point for all training (Online, Offline, PPO, BlendRL, IQL).
- `conf/`: Hydra configuration hierarchy.
  - `config.yaml`: Main defaults and SLURM settings.
  - `agent/`: Agent-specific hyperparameters (ppo, blendrl, iql, blendrl_iql).
  - `env/`: Environment-specific settings (seaquest, cartpole).
  - `mode/`: Execution modes (online, offline).
- `src/`: Core source code.
  - `agents/`: PyTorch Lightning implementations of RL algorithms.
  - `data/`: Data loading and dataset management.
- `results/`: Output directory for logs, checkpoints, and datasets.
  - `logs/`: CSV logs for easy plotting.
  - `tensorboard/`: Local TensorBoard visualizations.
  - `checkpoints/`: Model weights (best_model.pth).
  - `datasets/`: Transitions saved for offline training.
- `.legacy/`: Archive of the original scripts and results.

## Quick Start

### Local PPO Run
```bash
export PYTHONPATH=".:nsfr:neumann:in/envs/seaquest:in/envs/mountaincar:$PYTHONPATH"
python train.py env=cartpole agent=ppo experiment_id=test_run local=true
```

### Cluster Multirun (SLURM)
```bash
python train.py -m env=seaquest agent=ppo,blendrl seed=1,2,3
```

### Offline Training
```bash
python train.py mode=offline agent=iql mode.dataset_path=results/datasets/exp001/ppo
```
