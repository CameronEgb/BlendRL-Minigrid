import torch as th
import gymnasium as gym
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from typing import Sequence
import numpy as np

class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    name = "cartpole_poc"
    pred2action = {
        "left": 0,
        "right": 1,
    }
    pred_names: Sequence

    def __init__(
        self,
        mode: str,
        n_envs: int,
        render_mode="rgb_array",
        seed=None,
        **kwargs
    ):
        super().__init__(mode)
        self.n_envs = n_envs
        
        self.envs = []
        for i in range(n_envs):
            env = gym.make("CartPole-v1", render_mode=render_mode)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.Autoreset(env)
            self.envs.append(env)
            
        self.n_actions = 2
        self.n_raw_actions = 2
        self.n_objects = 2
        self.n_features = 4
        self.seed = seed

    def reset(self):
        logic_states = []
        neural_states = []
        seed_i = self.seed
        for env in self.envs:
            obs, info = env.reset(seed=seed_i)
            logic_state, neural_state = self.extract_logic_state(obs), self.extract_neural_state(obs)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            if seed_i is not None:
                seed_i += 1
        return th.stack(logic_states), th.stack(neural_states)

    def step(self, actions, is_mapped: bool = False):
        rewards = []
        truncations = []
        dones = []
        infos = []
        logic_states = []
        neural_states = []
        for i, env in enumerate(self.envs):
            action = actions[i]
            if hasattr(action, "item"):
                action = action.item()
            obs, reward, done, truncation, info = env.step(action)
            logic_state, neural_state = self.extract_logic_state(obs), self.extract_neural_state(obs)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)
        return (th.stack(logic_states), th.stack(neural_states)), rewards, dones, truncations, infos

    def extract_logic_state(self, obs):
        state = th.zeros((self.n_objects, self.n_features), dtype=th.float32)
        state[0] = th.tensor(obs, dtype=th.float32)
        state[1] = th.tensor(obs, dtype=th.float32)
        return state

    def extract_neural_state(self, obs):
        return th.tensor(obs, dtype=th.float32)


    def get_action_meanings(self):
        return ["left", "right"]

    def close(self):
        for env in self.envs:
            env.close()
