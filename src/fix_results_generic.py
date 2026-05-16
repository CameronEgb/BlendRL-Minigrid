import os
import pandas as pd
import yaml
import json
import shutil
from pathlib import Path
import argparse

def fix_results(exp_id, group=None):
    source_dir = Path("results/logs") / exp_id
    if group:
        target_dir = Path("results/experiments") / group / exp_id
    else:
        target_dir = Path("results/experiments") / exp_id

    if not source_dir.exists():
        print(f"Error: {source_dir} not found.")
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Map version to agent/mode to only keep the latest
    version_map = {}
    for version_dir in sorted(source_dir.glob("version_*"), key=lambda x: int(x.name.split('_')[1])):
        hparams_path = version_dir / "hparams.yaml"
        if hparams_path.exists():
            with open(hparams_path, 'r') as f:
                try:
                    hparams = yaml.safe_load(f)
                    if not hparams: continue
                    cfg = hparams.get('cfg', hparams)
                    agent = cfg.get('agent', {}).get('name', 'unknown')
                    # Handle both OmegaConf and dict
                    if isinstance(agent, dict): agent = agent.get('name', 'unknown')
                    
                    mode = cfg.get('mode', {}).get('type', 'unknown')
                    if isinstance(mode, dict): mode = mode.get('type', 'unknown')
                    
                    version_map[f"{agent}_{mode}_{version_dir.name}"] = version_dir
                except Exception as e:
                    print(f"Error reading {hparams_path}: {e}")

    for key, version_dir in version_map.items():
        metrics_path = version_dir / "metrics.csv"
        hparams_path = version_dir / "hparams.yaml"
        
        if metrics_path.exists():
            with open(hparams_path, 'r') as f:
                hparams = yaml.safe_load(f)
                cfg = hparams.get('cfg', hparams)
                
            run_name = key
            run_target = target_dir / run_name
            run_target.mkdir(parents=True, exist_ok=True)
            
            with open(run_target / "config.yaml", 'w') as f:
                yaml.dump(cfg, f)
                
            shutil.copy(metrics_path, run_target / "metrics.csv")
                
            df = pd.read_csv(metrics_path)
            results = []
            eval_df = df.dropna(subset=["eval/reward"])
            
            # Calculate transition count per epoch for online
            env_cfg = cfg.get('env', {})
            num_envs = env_cfg.get('num_envs', 4)
            num_steps = env_cfg.get('num_steps', 500)
            steps_per_epoch = num_envs * num_steps
            
            for i, (idx, row) in enumerate(eval_df.iterrows()):
                step = row['step']
                if mode == 'online':
                    # If step is small (e.g. < 1000), it's probably epoch index
                    # If it's large, it might already be transition count
                    if step < 1000:
                        limit = int(step * steps_per_epoch)
                    else:
                        limit = int(step)
                else:
                    limit = int(step)
                
                results.append({
                    "data_limit": limit,
                    "avg_reward": float(row['eval/reward']),
                    "std_reward": 0.0
                })
            
            if results:
                with open(run_target / "results.json", 'w') as f:
                    json.dump(results, f)

    print(f"Restructured results in {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=str, help="Experiment ID to fix")
    parser.add_argument("--group", type=str, default=None, help="Target group name")
    args = parser.parse_args()
    fix_results(args.exp_id, group=args.group)
