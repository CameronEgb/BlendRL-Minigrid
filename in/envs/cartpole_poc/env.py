import torch as th
import gymnasium as gym
from nudge.env import NudgeBaseEnv
from typing import Sequence
import numpy as np

class NudgeEnv(NudgeBaseEnv):
    name = "cartpole_poc"
    pred2action = {
        "left": 0,
        "right": 1,
    }
    pred_names: Sequence

    def __init__(
        self, mode: str, render_mode="rgb_array", seed=None, **kwargs
    ):
        super().__init__(mode)
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = gym.wrappers.Autoreset(self.env)
        
        self.n_actions = 2
        self.n_raw_actions = 2
        self.n_objects = 2 # [agent, env]
        self.n_features = 4 # [cart_pos, cart_vel, pole_angle, pole_vel]
        self.seed = seed

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        logic_state, neural_state = self.extract_logic_state(obs), self.extract_neural_state(obs)
        logic_state = logic_state.unsqueeze(0)
        return logic_state, neural_state

    def step(self, action, is_mapped: bool = False):
        if hasattr(action, "item"):
            action = action.item()
        obs, reward, done, truncation, info = self.env.step(action)
        logic_state, neural_state = self.extract_logic_state(obs), self.extract_neural_state(obs)
        logic_state = logic_state.unsqueeze(0)
        return (logic_state, neural_state), reward, done, truncation, info

    def extract_logic_state(self, obs):
        state = th.zeros((self.n_objects, self.n_features), dtype=th.float32)
        # Object 0: Agent
        state[0] = th.tensor(obs, dtype=th.float32)
        # Object 1: Env
        state[1] = th.tensor(obs, dtype=th.float32)
        return state

    def extract_neural_state(self, obs):
        logic_state = self.extract_logic_state(obs)
        return logic_state.view(-1).unsqueeze(0)

    def get_action_meanings(self):
        return ["left", "right"]

    def close(self):
        self.env.close()
