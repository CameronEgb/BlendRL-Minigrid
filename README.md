# BlendRL: Neural-Symbolic Reinforcement Learning Framework

A PyTorch Lightning & Hydra-powered framework for joint symbolic (logic) and neural policy learning, comparing Online vs. Offline learning efficiency across benchmark environments.

---

## 1. Directory Structure

```
├── run_pipeline.py                 # Primary entry point (orchestrates online/offline phases & sweeps)
├── src/                            # Core package source code
│   ├── train.py                    # PyTorch Lightning driver script
│   ├── methods/                    # RL agents (PPO, IQL, CQL, BlendRL, BlendRL-IQL, BlendRL-CQL)
│   ├── blendrl/                    # Environment vectorization & reasoner interfaces
│   ├── data/                       # Offline replay data modules & transition readers
│   ├── nsfr/                       # Neural-Symbolic Forward Reasoner engine
│   └── nudge/                      # Logic predicate & rule evaluation wrappers
├── in/                             # All input specifications & datasets
│   ├── config/                     # Hydra configuration system (agent, env, experiment, mode)
│   ├── datasets/                   # Static offline datasets (MIMIC, Pyrenees, generated replay buffers)
│   ├── envs/                       # Custom environment reward functions & MLP models
│   └── rules/                      # Symbolic logic rulesets (.prolog / .nudge)
├── plot/                           # Modular plotting framework
│   ├── manager.py                  # Auto-dispatcher called by run_pipeline.py
│   ├── convergence.py              # Convergence reward & episode length curves
│   ├── losses.py                   # Loss metric curves with per-metric filtering
│   └── reports.py                  # Markdown hyperparameter & comparison report generator
├── scripts/                        # Maintenance & pre-processing tools
│   ├── reorganize_results.py       # Reorganizes results/ folder to match experiment group specs
│   └── preprocess_pyrenees.py      # Pyrenees dataset conversion utility
└── results/                        # Central output storage (organized by group/experiment_id/agent)
    ├── logs/                       # CSV metrics & event logs
    ├── plots/                      # Generated visualization PNGs & Markdown reports
    ├── checkpoints/                # Model checkpoints saved at evaluation intervals
    ├── datasets/                   # Transitions saved during Phase 1 online runs
    ├── optuna/                     # SQLite Optuna databases for hyperparameter sweeps
    └── hydra/                      # Hydra stdout run logs (dir output redirects)
```

---

## 2. Quick Start

### Running an Experiment Pipeline Locally
Run an end-to-end experiment cycle (online generation $\rightarrow$ dataset saving $\rightarrow$ offline comparison $\rightarrow$ auto-plotting):

```bash
# Run CartPole benchmark experiment
python3 run_pipeline.py cp_final

# Run MIMIC offline comparison benchmark
python3 run_pipeline.py mimic_comparison

# Quick local sanity check with restricted timesteps
python3 run_pipeline.py cp_final total_timesteps=1000 intervals_count=2 eval_episodes=5 local=true
```

### Direct Model Training
To train a single agent using `src/train.py`:

```bash
# Train PPO on CartPole
python3 src/train.py +experiment=test_cp_fast mode=online agent=ppo/cp_tuned

# Train Offline IQL on CartPole replay buffer
python3 src/train.py +experiment=test_cp_fast mode=offline agent=iql/cp_tuned mode.dataset_path=in/datasets/cartpole/cp_final/ppo_cp_tuned
```

---

## 3. Modular Plotting (`plot/`)

Generate specific visualizations standalone for any experiment ID:

```bash
# Run auto-plotter manager (dispatches all plotters listed in experiment config)
python3 plot/manager.py cp_final

# Plot convergence curves standalone
python3 plot/convergence.py cp_final --window 15 --dpi 300

# Plot specific loss metrics standalone
python3 plot/losses.py cp_final --metrics losses/q_loss losses/actor_loss --window 20
```

---

## 4. Hyperparameter Sweeps (Optuna)

Run multi-trial sweeps locally or submit Slurm batch jobs to the cluster:

```bash
# Run hyperparameter sweep for all 4 MIMIC architectures (Neural, NeSy x2, Symbolic)
python3 run_pipeline.py tune_mimic_all -m

# Optuna SQLite databases are saved automatically in results/optuna/
```

---

## 5. Maintenance & Syncing

Reorganize your local or remote cluster results directory tree to match canonical experiment configurations:

```bash
python3 scripts/reorganize_results.py
```
