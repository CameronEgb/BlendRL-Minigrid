"""MIMIC environment hooks for pipeline customization."""
import os
import subprocess
import sys
from pathlib import Path


class Hooks:
    """MIMIC-specific pipeline hooks."""
    
    @staticmethod
    def transform_rewards(reader, cfg):
        """Rewrite rewards based on MIMIC_REWARD_TYPE (behavioral/tqn/outcome)."""
        import torch
        import numpy as np
        
        reward_type = os.environ.get("MIMIC_REWARD_TYPE", "tqn").lower()
        
        # Only apply to MIMIC-shaped observations (46 features)
        if reader.obs.shape[-1] != 46:
            return
        
        if reward_type == "behavioral" and reader.obs.shape[-1] == 46:
            print("MIMIC_REWARD_TYPE=behavioral detected! Zeroing out clinician penalties for death cases in memory...")
            reader.rewards[reader.rewards < 0.0] = 0.0
        elif reward_type == "tqn" and reader.obs.shape[-1] == 46:
            print("MIMIC_REWARD_TYPE=tqn detected! Recomputing TQN stage-severity & action-cost rewards in memory...")
            import sys, importlib
            if "src" not in sys.path:
                sys.path.append("src")
            mimic_env_mod = importlib.import_module("in.envs.mimic.env_vectorized")
            compute_sev = mimic_env_mod.compute_tqn_stage_severity
            compute_cost = mimic_env_mod.compute_tqn_action_cost

            n_transitions = len(reader.obs)
            new_rewards = reader.rewards.clone().float()
            start_idx = 0

            for idx in range(n_transitions):
                obs_curr = reader.obs[idx].numpy()
                curr_sev = compute_sev(obs_curr)
                if idx > start_idx:
                    obs_prev = reader.obs[idx - 1].numpy()
                    prev_sev = compute_sev(obs_prev)
                else:
                    prev_sev = 0.0

                act = int(reader.actions[idx].item())
                act_cost = compute_cost(act, obs_curr)
                new_rewards[idx] = (curr_sev - prev_sev) - act_cost

                if reader.dones[idx] == 1.0:
                    start_idx = idx + 1
            reader.rewards = new_rewards
        elif reward_type == "outcome" and reader.obs.shape[-1] == 46:
            print("MIMIC_REWARD_TYPE=outcome detected! Recomputing offline dataset rewards in memory...")
            n_transitions = len(reader.obs)
            new_rewards = reader.rewards.clone().float()
            
            start_idx = 0
            for idx in range(n_transitions):
                if reader.dones[idx] == 1.0 or idx == n_transitions - 1:
                    # Trajectory goes from start_idx to idx (inclusive)
                    # Determine outcome: if original reward at the end is negative, the patient died (1)
                    outcome = 1.0 if reader.rewards[idx] < 0.0 else 0.0
                    
                    # Recompute rewards for this trajectory
                    for step_idx in range(start_idx, idx + 1):
                        reward_t = 0.0
                        if step_idx == idx:
                            reward_t = 15.0 if outcome == 0.0 else -15.0
                            
                        new_rewards[step_idx] = reward_t
                    
                    start_idx = idx + 1
            reader.rewards = new_rewards
        elif reward_type == "ep_shaped" and reader.obs.shape[-1] == 46:
            # Reciprocal Refinement: TQN base + EP potential-based reward shaping.
            # First apply TQN rewards as the base signal, then overlay EP-based
            # potential shaping: r = r_TQN + λ*(γ*Φ(s') - Φ(s)), Φ = -P_EP(shock).
            # See Ng, Harada, Russell (1999) — provably preserves optimal policies.
            print("MIMIC_REWARD_TYPE=ep_shaped detected! Applying TQN base + EP potential shaping...")
            
            # Step 1: Apply TQN base rewards
            import importlib
            if "src" not in sys.path:
                sys.path.append("src")
            mimic_env_mod = importlib.import_module("in.envs.mimic.env_vectorized")
            compute_sev = mimic_env_mod.compute_tqn_stage_severity
            compute_cost = mimic_env_mod.compute_tqn_action_cost

            n_transitions = len(reader.obs)
            new_rewards = reader.rewards.clone().float()
            start_idx = 0
            for idx in range(n_transitions):
                obs_curr = reader.obs[idx].numpy()
                curr_sev = compute_sev(obs_curr)
                if idx > start_idx:
                    obs_prev = reader.obs[idx - 1].numpy()
                    prev_sev = compute_sev(obs_prev)
                else:
                    prev_sev = 0.0
                act = int(reader.actions[idx].item())
                act_cost = compute_cost(act, obs_curr)
                new_rewards[idx] = (curr_sev - prev_sev) - act_cost
                if reader.dones[idx] == 1.0:
                    start_idx = idx + 1
            reader.rewards = new_rewards
            
            # Step 2: Overlay EP potential-based shaping
            from src.reward_shaping import shape_rewards_ep
            shape_rewards_ep(reader, cfg)
    
    @staticmethod
    def preprocess_dataset(cfg):
        """Auto-convert .npz files to .pkl chunks if needed."""
        target_npz = cfg.env.get("dataset_name", "mimic_lazy_0_interventions_balanced.npz")
        npz_stem = Path(target_npz).stem
        target_dir = Path("in/datasets/mimic") / npz_stem
        cql_dir = Path("in/datasets/mimic/cql")
        if not ((target_dir.exists() and any(target_dir.glob("*.pkl"))) or (cql_dir.exists() and any(cql_dir.glob("*.pkl")))):
            print(f"\n=== Auto-Converting MIMIC NPZ Dataset ({target_npz}) to PKL Format ===")
            subprocess.run([sys.executable, "scripts/convert_npz_to_pkl.py", target_npz], check=True)
    
    @staticmethod
    def post_training_eval(cfg, local_val):
        """Run early prediction evaluation after MIMIC training."""
        if not local_val:
            return
            
        print("\n=== Phase: Local Early Prediction Evaluation ===")
        ckpt_dir_root = Path("results/checkpoints") / cfg.group / cfg.experiment_id
        if ckpt_dir_root.exists():
            if any(ckpt_dir_root.rglob("best_model*.ckpt")):
                # In hooks, we import the helper from run_pipeline or execute script
                # Since we just extract logic, we need to run eval.py directly or via import
                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.abspath("src") + ":" + env.get("PYTHONPATH", "")
                venv_python = sys.executable  # Simplification for hooks since it's already in venv
                
                # Check for args.remake equiv? The prompt says "COPY the mimic-specific eval trigger"
                # We'll just run it simply or try to do exactly what run_early_prediction_eval did.
                cmd = [
                    venv_python, "src/early_prediction/eval.py",
                    "--checkpoint", str(ckpt_dir_root),
                    "--ep-ckpt-root", "results/checkpoints/early_prediction",
                ]
                # Try to get remake from somewhere if needed, but not strictly possible from just cfg and local_val.
                print(f"\n=== Running Early Prediction Evaluation for checkpoint: {ckpt_dir_root} ===")
                subprocess.run(cmd, check=True, env=env)
            else:
                print(f"Warning: No best_model*.ckpt files found under {ckpt_dir_root} for evaluation.")
        else:
            print(f"Warning: Checkpoint root directory {ckpt_dir_root} does not exist.")
