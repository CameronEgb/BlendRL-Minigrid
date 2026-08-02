import numpy as np
import os
import sys
import pickle
from pathlib import Path

# Fix path to load src.dataset_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset_utils import DatasetWriter

# Paths
datasets_dir = os.environ.get("MIMIC_DATASET_DIR", "")
if not datasets_dir or not os.path.exists(datasets_dir):
    for candidate in [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../in/datasets/mimic")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../in/datasets")),
        os.path.abspath(os.path.join(os.getcwd(), "in/datasets/mimic")),
        os.path.abspath(os.path.join(os.getcwd(), "in/datasets")),
        "/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2",
        "/mnt/beegfs/cegbert/NeSyRL/in/datasets/mimic",
        "/mnt/beegfs/cegbert/NeSyRL/in/datasets",
        "/mnt/beegfs/cegbert/MIMIC 2"
    ]:
        if os.path.exists(candidate):
            datasets_dir = candidate
            break

# Select NPZ file (default to the full balanced dataset)
npz_filename = os.environ.get("MIMIC_DATASET_NAME", "mimic_lazy_0_interventions_balanced.npz")
if len(sys.argv) > 1:
    npz_filename = sys.argv[1]

npz_path = os.path.join(datasets_dir, npz_filename)
if not os.path.exists(npz_path):
    # Fallback to alternate filenames if default doesn't exist
    for alt in ["mimic_lazy_0_interventions_balanced.npz", "mimic_expert_demonstrations.npz"]:
        alt_p = os.path.join(datasets_dir, alt)
        if os.path.exists(alt_p):
            npz_path = alt_p
            break

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../in/datasets/mimic/cql"))

if not os.path.exists(npz_path):
    print(f"Error: NPZ dataset not found at {npz_path}")
    sys.exit(1)

print(f"Loading NPZ dataset from {npz_path}...")
data = np.load(npz_path, allow_pickle=True)
X = data['X']  # (N, 240, 49)
y = data['y']  # (N, 1)
mask = data['mask']  # (N, 240, 1)

writer = DatasetWriter(save_dir=out_dir, chunk_size=200000, env_name="mimic")

print("Converting trajectories to transitions...")
total_transitions = 0

for i in range(len(X)):
    mask_patient = mask[i].squeeze()
    active_steps = np.where(mask_patient != -1)[0]
    
    if len(active_steps) == 0:
        continue
        
    T = len(active_steps)
    outcome = y[i, 0]
    
    for t_idx, t in enumerate(active_steps):
        obs = X[i, t, :46]
        logic_obs = np.stack([obs, obs], axis=0) # Duplicate for nsfr reasoner format
        
        # Action is Antibiotics (index 47)
        action = int(X[i, t, 47])
        
        # Default behavioral reward (overridden in-memory if MIMIC_REWARD_TYPE=outcome)
        reward = (1.0 / T) if action == int(X[i, t, 47]) else 0.0
        if outcome != 0:
            reward = -reward # Negative for death
            
        if t_idx == len(active_steps) - 1:
            # Terminal step
            next_obs = obs.copy()
            next_logic_obs = logic_obs.copy()
            done = True
        else:
            t_next = active_steps[t_idx + 1]
            next_obs = X[i, t_next, :46]
            next_logic_obs = np.stack([next_obs, next_obs], axis=0)
            done = False
            
        writer.add(
            obs=obs,
            logic_obs=logic_obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            next_logic_obs=next_logic_obs,
            done=done
        )
        total_transitions += 1

writer.close()
print(f"Successfully converted {len(X)} patients and saved {total_transitions} transitions to {out_dir}")
