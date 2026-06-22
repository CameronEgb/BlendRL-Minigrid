import pandas as pd
import glob
import os

def get_best_results(agent_path):
    files = glob.glob(os.path.join(agent_path, "**/metrics.csv"), recursive=True)
    best_overall = -float('inf')
    best_file = ""
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'eval/reward' in df.columns:
                max_reward = df['eval/reward'].max()
                if max_reward > best_overall:
                    best_overall = max_reward
                    best_file = f
        except Exception:
            continue
    return best_overall, best_file

ppo_best, ppo_file = get_best_results("results/logs/tuning/cp_final_tune/ppo_cp_tuned")
blend_best, blend_file = get_best_results("results/logs/tuning/cp_final_tune/blendrl_cp_tuned")

print(f"PPO Best Reward: {ppo_best}")
print(f"PPO Best File: {ppo_file}")
print(f"BlendRL Best Reward: {blend_best}")
print(f"BlendRL Best File: {blend_file}")
