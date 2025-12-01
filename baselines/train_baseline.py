import argparse
import os
import time
import csv
import heapq
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from minigrid.envs import DynamicObstaclesEnv
from minigrid.core.constants import DIR_TO_VEC
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- Custom CNN for MiniGrid (Fixes 7x7 input size issue) ---
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW input (SB3 handles the transpose)
        n_input_channels = observation_space.shape[0]
        
        # A small CNN architecture for 7x7 grids
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# --- Cardinal Action Wrapper ---
class CardinalActionWrapper(gym.ActionWrapper):
    """
    Changes the action space to:
    0: Move Up (North)
    1: Move Right (East)
    2: Move Down (South)
    3: Move Left (West)
    
    It achieves this by instantly setting the agent's direction 
    and then executing a 'forward' action.
    """
    def __init__(self, env):
        super().__init__(env)
        # 0=North, 1=East, 2=South, 3=West (Clockwise from Top)
        self.action_space = spaces.Discrete(4)
        
        # Map our actions (0-3) to MiniGrid direction constants
        # MiniGrid: 0=East, 1=South, 2=West, 3=North
        self.action_to_dir = {
            0: 3, # Up -> North
            1: 0, # Right -> East
            2: 1, # Down -> South
            3: 2  # Left -> West
        }

    def action(self, action):
        # This method is for transforming actions passed to step(), 
        # but since we need to modify state (agent_dir) *before* the step,
        # we override step() directly below instead.
        return action

    def step(self, action):
        # 1. Force agent direction
        target_dir = self.action_to_dir[action]
        self.env.unwrapped.agent_dir = target_dir
        
        # 2. Execute 'Move Forward' (MiniGrid action 2)
        return self.env.step(2)

# --- Custom Environment Factory ---
def make_custom_env(n_obstacles, size=8, render_mode=None, rank=0, seed=0):
    def _init():
        env = DynamicObstaclesEnv(
            size=size,
            n_obstacles=n_obstacles,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            render_mode=render_mode
        )
        # 1. Apply Cardinal Wrapper (Up/Down/Left/Right)
        env = CardinalActionWrapper(env)
        
        # 2. Apply Image Wrapper for CNNs
        env = ImgObsWrapper(env)
        
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init

# --- Updated A* Oracle (Cardinal) ---
class CardinalAStar:
    def __init__(self, env):
        self.env = env
        
    def heuristic(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_action(self):
        grid = self.env.unwrapped.grid
        start_pos = self.env.unwrapped.agent_pos
        goal_pos = None

        # Find goal
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj and obj.type == 'goal':
                    goal_pos = (x, y)
                    break
        
        if not goal_pos: return self.env.action_space.sample()

        # State: (x, y) - No direction needed now!
        start_node = tuple(start_pos)
        
        # Priority Queue: (cost, current_node, first_action_to_take)
        # We store 'first_action' to know which move started the path
        queue = [(0, start_node, None)]
        visited = set()
        
        while queue:
            cost, current, first_action = heapq.heappop(queue)
            cx, cy = current
            
            if (cx, cy) == goal_pos:
                return first_action if first_action is not None else 0
            
            if current in visited:
                continue
            visited.add(current)
            
            # Neighbors: Up(0), Right(1), Down(2), Left(3)
            # Corresponding deltas (x, y):
            # Up (North): (0, -1)
            # Right (East): (1, 0)
            # Down (South): (0, 1)
            # Left (West): (-1, 0)
            moves = [
                (0, 0, -1),
                (1, 1, 0),
                (2, 0, 1),
                (3, -1, 0)
            ]
            
            for action_idx, dx, dy in moves:
                nx, ny = cx + dx, cy + dy
                
                # Check bounds and obstacles
                # Note: Treating dynamic obstacles as static for this planning step
                if 0 <= nx < grid.width and 0 <= ny < grid.height:
                    cell = grid.get(nx, ny)
                    is_blocked = False
                    if cell:
                        if cell.type in ['wall', 'ball']: 
                            is_blocked = True
                    
                    if not is_blocked and (nx, ny) not in visited:
                        h = self.heuristic((nx, ny), goal_pos)
                        # If this is the start node, set the action, else keep existing
                        next_action = action_idx if first_action is None else first_action
                        heapq.heappush(queue, (cost + 1 + h, (nx, ny), next_action))
                
        # Fallback: No path found (surrounded), wait/random
        return self.env.action_space.sample()

def run_heuristic_agent(agent_type, args):
    log_dir = f"./baselines/logs/{agent_type}_obst{args.obstacles}_{int(time.time())}/"
    os.makedirs(log_dir, exist_ok=True)
    print(f"--> Running {agent_type} baseline (Cardinal Actions)...")
    
    # Create single environment
    env = make_custom_env(n_obstacles=args.obstacles, size=8)()
    
    # Create Monitor CSV
    monitor_path = os.path.join(log_dir, "0.monitor.csv")
    with open(monitor_path, "w") as f:
        f.write(f"# {{ 't_start': {time.time()}, 'env_id': 'MiniGrid-Dynamic-Cardinal' }}\n")
        f.write("r,l,t\n")

    obs, _ = env.reset()
    total_steps = 0
    episode_reward = 0
    episode_len = 0
    
    astar = CardinalAStar(env) if agent_type == 'astar' else None

    start_time = time.time()
    
    while total_steps < args.steps:
        if agent_type == 'random':
            action = env.action_space.sample()
        elif agent_type == 'astar':
            action = astar.get_action()

        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_len += 1
        total_steps += 1
        
        if terminated or truncated:
            with open(monitor_path, "a") as f:
                f.write(f"{episode_reward},{episode_len},{time.time() - start_time}\n")
            
            if total_steps % 10000 == 0:
                print(f"Step {total_steps}/{args.steps} | Last Reward: {episode_reward:.2f}")

            obs, _ = env.reset()
            episode_reward = 0
            episode_len = 0

    print(f"--> {agent_type} baseline finished.")

def train(args):
    if args.algo in ['random', 'astar']:
        run_heuristic_agent(args.algo, args)
        return

    # 1. Setup Logging
    log_dir = f"./baselines/logs/{args.algo}_obst{args.obstacles}_{int(time.time())}/"
    os.makedirs(log_dir, exist_ok=True)
    print(f"--> Training {args.algo} (Cardinal) with {args.obstacles} obstacles.")
    
    # 2. Setup Environment
    n_envs = 4 if args.algo.lower() in ['ppo', 'a2c'] else 1
    env = make_vec_env(
        make_custom_env(n_obstacles=args.obstacles, size=8),
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv
    )

    # 3. Setup Model
    tensorboard_log = "./tensorboard_logs/"
    
    # Define policy kwargs to use our custom CNN
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    if args.algo.lower() == 'ppo':
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log,
                    learning_rate=0.0003, n_steps=2048, batch_size=64, ent_coef=0.01,
                    policy_kwargs=policy_kwargs)
    elif args.algo.lower() == 'a2c':
        model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log,
                    learning_rate=0.0007, n_steps=5, ent_coef=0.01,
                    policy_kwargs=policy_kwargs)
    elif args.algo.lower() == 'dqn':
        model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log,
                    learning_rate=0.0001, buffer_size=100000, learning_starts=1000,
                    target_update_interval=1000, train_freq=4, gradient_steps=1,
                    exploration_fraction=0.1, exploration_final_eps=0.05,
                    policy_kwargs=policy_kwargs)
    else:
        raise ValueError("Unknown algorithm")

    # 4. Callbacks
    # Ensure eval_env is a VecEnv (Wrapped with TransposeImage for CNNs)
    eval_env = make_vec_env(
        make_custom_env(n_obstacles=args.obstacles, size=8),
        n_envs=1,
        seed=args.seed + 1000, 
        vec_env_cls=DummyVecEnv
    )
    
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=log_dir, log_path=log_dir, 
        eval_freq=max(10000 // n_envs, 1), deterministic=True, render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1), save_path=log_dir, name_prefix=f"{args.algo}_model"
    )

    # 5. Train
    model.learn(total_timesteps=args.steps, callback=[eval_callback, checkpoint_callback], progress_bar=True)
    model.save(f"{log_dir}/final_model")
    print(f"--> Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baselines for MiniGrid Dynamic Obstacles (Cardinal)")
    parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'dqn', 'a2c', 'random', 'astar'])
    parser.add_argument("--obstacles", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.obstacles > 20:
        print("Warning: High number of obstacles may make the environment unsolvable.")
        
    train(args)