from typing import Sequence
import torch as th
import torch.nn.functional as F
import gymnasium as gym

from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.world_object import Goal, Wall


class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    """
    Vectorized MiniGrid environment for BlendRL.

    Each env:
        logic_state: (4,4)
        neural_state: (4,84,84)

    Batch:
        logic_states:  (n_envs, 4,4)
        neural_states: (n_envs, 4,84,84)
    """

    name = "minigrid"

    pred2action = {
        "move_left": 0,
        "move_right": 1,
        "move_forward": 2,
        "turn_left": 3,
        "turn_right": 4,
        "done": 6,
    }
    pred_names: Sequence

    def __init__(self, mode: str, n_envs: int,
                 render_mode="rgb_array", render_oc_overlay=False, seed=None):
        super().__init__(mode)

        self.n_envs = n_envs
        self.seed = seed
        self.render_mode = render_mode

        self.n_objects = 4
        self.n_features = 4

        self.n_actions = 7
        self.n_raw_actions = 7

        self.envs = []
        for i in range(n_envs):
            env = gym.make("MiniGrid-Empty-5x5-v0", render_mode=render_mode)
            env = FullyObsWrapper(env)
            self.envs.append(env)

    def reset(self):
        logic_states = []
        neural_states = []

        seed_i = self.seed

        for env in self.envs:
            if seed_i is not None:
                obs, _ = env.reset(seed=seed_i)
                seed_i += 1
            else:
                obs, _ = env.reset()

            img = th.tensor(obs["image"], dtype=th.float32)

            logic_state = self.extract_logic_state_objects(env)
            neural_state = self.extract_neural_state(img)

            logic_states.append(logic_state)
            neural_states.append(neural_state)

        return th.stack(logic_states), th.stack(neural_states)

    def step(self, actions, is_mapped: bool = False):
        rewards = []
        truncations = []
        dones = []
        infos = []
        logic_states = []
        neural_states = []

        for i, env in enumerate(self.envs):
            action = int(actions[i])

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            img = th.tensor(obs["image"], dtype=th.float32)

            logic_state = self.extract_logic_state_objects(env)
            neural_state = self.extract_neural_state(img)

            logic_states.append(logic_state)
            neural_states.append(neural_state)
            rewards.append(reward)
            truncations.append(truncated)
            dones.append(done)
            infos.append(info)

        return (
            (th.stack(logic_states), th.stack(neural_states)),
            rewards,
            truncations,
            dones,
            infos,
        )

    def extract_logic_state_objects(self, env) -> th.Tensor:
        uenv = env.unwrapped

        ax, ay = uenv.agent_pos
        ad = uenv.agent_dir

        gx, gy = 0, 0
        found_goal = False
        for x in range(uenv.width):
            for y in range(uenv.height):
                obj = uenv.grid.get(x, y)
                if isinstance(obj, Goal):
                    gx, gy = x, y
                    found_goal = True
                    break
            if found_goal:
                break

        wx, wy = 0, 0
        found_wall = False
        for x in range(uenv.width):
            for y in range(uenv.height):
                obj = uenv.grid.get(x, y)
                if isinstance(obj, Wall):
                    wx, wy = x, y
                    found_wall = True
                    break
            if found_wall:
                break

        logic = th.tensor(
            [
                [0, 0, 0, 0],
                [ax, ay, ad, 1],
                [gx, gy, 0, 1],
                [wx, wy, 0, 1],
            ],
            dtype=th.int32,
        )
        return logic

    def extract_neural_state(self, img: th.Tensor) -> th.Tensor:
        assert img.numel() == 75, f"Expected 75 elements (5x5x3), got {img.numel()}"

        x = img.permute(2, 0, 1).unsqueeze(0)
        gray = x.mean(dim=1, keepdim=True)
        gray_84 = F.interpolate(gray, size=(84, 84), mode="nearest")
        stacked = gray_84.repeat(1, 4, 1, 1)
        return stacked.squeeze(0)

    def close(self):
        for env in self.envs:
            env.close()
