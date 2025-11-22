from typing import Sequence
import torch as th
import torch.nn.functional as F
import gymnasium as gym

from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.world_object import Goal, Wall, Ball


class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    """
    Vectorized MiniGrid environment for BlendRL.

    look at documentation in env.py for info on matching methods
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
                 render_mode="rgb_array", render_oc_overlay=False, seed=None,num_balls=None):
        super().__init__(mode)

        self.n_envs = n_envs
        self.seed = seed
        self.render_mode = render_mode
        self.num_balls = num_balls

        env_kwargs = {}
        if self.num_balls is not None:
            env_kwargs["n_obstacles"] = self.num_balls
        self.n_objects = 5
        self.n_features = 4

        self.n_actions = 7
        self.n_raw_actions = 7

        self.envs = []
        for i in range(n_envs):
            env = gym.make("MiniGrid-Dynamic-Obstacles-6x6-v0", render_mode=render_mode,**env_kwargs)
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

        # --- ENEMIES: nearest enemy summary for this env ---
        enemy_positions = []
        if hasattr(uenv, "obstacles") and uenv.obstacles is not None:
            enemy_positions.extend([tuple(obj.cur_pos) for obj in uenv.obstacles])

        if not enemy_positions:
            for x in range(uenv.width):
                for y in range(uenv.height):
                    obj = uenv.grid.get(x, y)
                    if isinstance(obj, Ball):
                        enemy_positions.append((x, y))

        if enemy_positions:
            dists = [
                (abs(ax - ex) + abs(ay - ey), ex, ey)
                for (ex, ey) in enemy_positions
            ]
            dists.sort()
            _, ex, ey = dists[0]
            enemy_row = [ex, ey, 0, 1]
        else:
            enemy_row = [-1, -1, 0, 1]

        logic = th.tensor(
            [
                [0, 0, 0, 0],
                [ax, ay, ad, 1],
                [gx, gy, 0, 1],
                [wx, wy, 0, 1],
                enemy_row,
            ],
            dtype=th.int32,
        )
        return logic

    def extract_neural_state(self, img: th.Tensor) -> th.Tensor:
        #assert img.numel() == 75, f"Expected 75 elements (5x5x3), got {img.numel()}"

        x = img.permute(2, 0, 1).unsqueeze(0)
        gray = x.mean(dim=1, keepdim=True)
        gray_84 = F.interpolate(gray, size=(84, 84), mode="nearest")
        stacked = gray_84.repeat(1, 4, 1, 1)
        return stacked.squeeze(0)

    def close(self):
        for env in self.envs:
            env.close()
