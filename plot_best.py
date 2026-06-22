import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import pandas as pd
from collections import defaultdict

def moving_average(a, n=5):
    if len(a) == 0: return np.array([])
    n = min(len(a), n) if len(a) > 0 else 1
    a_padded = np.pad(a, (n-1, 0), mode='edge')
    ret = np.cumsum(a_padded, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def load_run_data(run_folder):
    data = {}
    metrics_path = run_folder / "metrics.csv"
    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            if df.empty: return None
            
            if "transitions" in df.columns:
                df['transitions'] = df['transitions'].ffill().bfill()
            elif "step" in df.columns:
                df['transitions'] = df['step']
            
            for col in df.columns:
                if col in ["step", "transitions", "epoch"]: continue
                subset = df[df[col].notna()].copy()
                if not subset.empty:
                    data[col] = {
                        "values": subset[col].tolist(),
                        "transitions": subset["transitions"].tolist() if "transitions" in subset.columns else []
                    }
            return data
        except:
            return None
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    args = parser.parse_args()

    base_dir = Path("results/logs/tuning") / args.experiment
    save_dir = Path("results/plots/tuning") / args.experiment / "best_only"
    save_dir.mkdir(parents=True, exist_ok=True)

    agent_folders = [f for f in base_dir.iterdir() if f.is_dir()]
    
    plt.figure(figsize=(10, 6))
    
    for agent_folder in agent_folders:
        versions = list(agent_folder.glob("version_*"))
        best_reward = -float('inf')
        best_data = None
        best_version = ""

        print(f"Analyzing {agent_folder.name}...")
        for v in versions:
            metrics_path = v / "metrics.csv"
            if metrics_path.exists():
                try:
                    df = pd.read_csv(metrics_path)
                    if 'eval/reward' in df.columns:
                        max_r = df['eval/reward'].max()
                        if max_r > best_reward:
                            best_reward = max_r
                            best_version = v.name
                            best_data = load_run_data(v)
                except:
                    continue
        
        if best_data and 'eval/reward' in best_data:
            d = best_data['eval/reward']
            x = d['transitions']
            y = d['values']
            plt.plot(x, y, label=f"{agent_folder.name} (Best: {best_version})")
            print(f"  Best for {agent_folder.name}: {best_version} (Reward: {best_reward})")

    plt.title(f"Best Trial Performance: {args.experiment}")
    plt.xlabel("Transitions")
    plt.ylabel("Eval Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "best_eval_reward.png")
    print(f"\nSaved best trial plot to: {save_dir / 'best_eval_reward.png'}")

if __name__ == "__main__":
    main()
