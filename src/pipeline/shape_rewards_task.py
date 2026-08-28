from src.pipeline.task_registry import register_task
from src.dataset_utils import DatasetReader, DatasetWriter
import os
import torch
from pathlib import Path
import logging
import sys

log = logging.getLogger(__name__)

@register_task("shape_rewards")
def run_shape_rewards(cfg, args, context):
    """
    Offline task to shape or transform rewards in a dataset.
    This replaces the old on-the-fly 'transform_rewards' hook.
    
    Expected config overrides:
    +input_dataset=in/datasets/mimic/some_dataset
    +output_dataset=in/datasets/mimic/some_dataset_shaped
    +reward_type=outcome  # or 'ep_shaped', 'behavioral', 'tqn'
    """
    input_path = cfg.get("input_dataset", None)
    output_path = cfg.get("output_dataset", None)
    reward_type = str(cfg.get("reward_type", "outcome")).lower()
    
    if not input_path or not output_path:
        log.error("Must provide +input_dataset and +output_dataset overrides.")
        sys.exit(1)
        
    log.info(f"Loading dataset from {input_path}...")
    reader = DatasetReader(input_path)
    
    if reader.obs.shape[-1] != 46:
        log.error("This task currently only supports MIMIC datasets (46 features).")
        sys.exit(1)
        
    n_transitions = len(reader.obs)
    log.info(f"Transforming rewards using type: {reward_type} on {n_transitions} transitions...")
    
    if reward_type == "behavioral":
        reader.rewards[reader.rewards < 0.0] = 0.0
        
    elif reward_type == "tqn":
        import importlib
        if "src" not in sys.path:
            sys.path.append("src")
        mimic_env_mod = importlib.import_module("in.envs.mimic.env_vectorized")
        compute_sev = mimic_env_mod.compute_tqn_stage_severity
        compute_cost = mimic_env_mod.compute_tqn_action_cost

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
        
    elif reward_type == "outcome":
        new_rewards = reader.rewards.clone().float()
        start_idx = 0
        for idx in range(n_transitions):
            if reader.dones[idx] == 1.0 or idx == n_transitions - 1:
                outcome = 1.0 if reader.rewards[idx] < 0.0 else 0.0
                for step_idx in range(start_idx, idx + 1):
                    reward_t = 0.0
                    if step_idx == idx:
                        reward_t = 15.0 if outcome == 0.0 else -15.0
                    new_rewards[step_idx] = reward_t
                start_idx = idx + 1
        reader.rewards = new_rewards
        
    elif reward_type == "ep_shaped":
        import importlib
        if "src" not in sys.path:
            sys.path.append("src")
        mimic_env_mod = importlib.import_module("in.envs.mimic.env_vectorized")
        compute_sev = mimic_env_mod.compute_tqn_stage_severity
        compute_cost = mimic_env_mod.compute_tqn_action_cost

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
        
        from src.reward_shaping import shape_rewards_ep
        shape_rewards_ep(reader, cfg)
        
    else:
        log.error(f"Unknown reward type: {reward_type}")
        sys.exit(1)
        
    log.info(f"Saving transformed dataset to {output_path}...")
    writer = DatasetWriter(output_path, env_name=cfg.env.name, seed=cfg.seed, max_transitions_per_file=100000)
    
    has_logic = getattr(reader, 'logic_obs', None) is not None
    
    obs_np = reader.obs.numpy()
    actions_np = reader.actions.numpy()
    rewards_np = reader.rewards.numpy()
    dones_np = reader.dones.numpy()
    next_obs_np = reader.next_obs.numpy()
    
    if has_logic:
        logic_obs_np = reader.logic_obs.numpy()
        next_logic_obs_np = reader.next_logic_obs.numpy()
        
    for i in range(n_transitions):
        writer.write(
            obs=obs_np[i],
            logic_obs=logic_obs_np[i] if has_logic else None,
            action=actions_np[i],
            reward=rewards_np[i],
            done=dones_np[i],
            next_obs=next_obs_np[i],
            next_logic_obs=next_logic_obs_np[i] if has_logic else None
        )
        
    writer.close()
    log.info(f"Dataset successfully saved to {output_path}!")
