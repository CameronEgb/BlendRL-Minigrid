import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from src.dataset_utils import DatasetWriter

def preprocess_pyrenees():
    data_dir = Path("in/datasets/pyrenees/Pyrenees data clean")
    out_cql_dir = Path("in/datasets/pyrenees/cql")
    out_cql_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted([f for f in glob.glob(str(data_dir / "*.csv")) if "problem.csv" not in f])
    print(f"Found {len(csv_files)} exercise CSV files.")

    meta_cols = [
        'feature_recordID', 'answerID', 'time', 'userID', 'problem',
        'decisionID', 'decisionPoint', 'decisionOrdering', 'substepMode',
        'KC', 'session', 'substepOrdering', 'action', 'reward'
    ]

    df_sample = pd.read_csv(csv_files[0], nrows=5)
    feat_cols = [c for c in df_sample.columns if c not in meta_cols]
    print(f"Identified {len(feat_cols)} feature columns.")

    print("Pass 1: Reading CSV files...")
    all_dfs = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows in dataset: {len(full_df)}")

    # Standardize numerical features
    feat_matrix = full_df[feat_cols].values.astype(np.float32)
    mean = np.mean(feat_matrix, axis=0)
    std = np.std(feat_matrix, axis=0)
    std[std == 0.0] = 1.0

    np.savez("in/datasets/pyrenees/pyrenees_scaler.npz", mean=mean, std=std, feat_cols=np.array(feat_cols))

    print("Pass 2: Creating trajectories & batch-writing dataset...")
    writer = DatasetWriter(save_dir=out_cql_dir, chunk_size=100000, env_name="pyrenees")

    npz_states = []
    npz_actions = []
    npz_rewards = []
    npz_dones = []

    total_transitions = 0
    total_trajectories = 0

    for df in all_dfs:
        norm_feats = (df[feat_cols].values.astype(np.float32) - mean) / std
        df_actions = df['action'].values.astype(np.int64)
        df_rewards = df['reward'].values.astype(np.float32)

        # Group by (userID, problem)
        grouped = df.groupby(['userID', 'problem'], sort=False).indices

        for (user, prob), indices in grouped.items():
            if len(indices) == 0:
                continue

            traj_states = norm_feats[indices]
            traj_actions = df_actions[indices]
            traj_rewards = df_rewards[indices]
            traj_len = len(indices)

            traj_dones = np.zeros(traj_len, dtype=np.float32)
            traj_dones[-1] = 1.0

            # Next obs
            traj_next_states = np.empty_like(traj_states)
            traj_next_states[:-1] = traj_states[1:]
            traj_next_states[-1] = traj_states[-1]

            # Fast batch add
            writer.batch_add(
                obs=traj_states,
                logic_obs=None,
                action=traj_actions,
                reward=traj_rewards,
                next_obs=traj_next_states,
                next_logic_obs=None,
                done=traj_dones
            )
            total_transitions += traj_len

            npz_states.append(traj_states)
            npz_actions.append(traj_actions)
            npz_rewards.append(traj_rewards)
            npz_dones.append(traj_dones)
            total_trajectories += 1

    writer.close()
    print(f"DatasetWriter saved {total_transitions} transitions in chunked .pkl format to {out_cql_dir}.")

    npz_path = Path("in/datasets/pyrenees/pyrenees_clean.npz")
    np.savez_compressed(
        npz_path,
        states=np.array(npz_states, dtype=object),
        actions=np.array(npz_actions, dtype=object),
        rewards=np.array(npz_rewards, dtype=object),
        dones=np.array(npz_dones, dtype=object)
    )
    print(f"Saved compressed evaluation dataset ({total_trajectories} trajectories) to {npz_path}.")
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_pyrenees()
