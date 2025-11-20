import torch as th
import torch.nn.functional as F
import gymnasium as gym

from nudge.env import NudgeBaseEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.world_object import Goal, Wall, Ball


class NudgeEnv(NudgeBaseEnv):
    """
    MiniGrid NudgeEnv wrapper for BlendRL.

    Env: MiniGrid-Empty-5x5-v0 with FullyObsWrapper

    Logic state (object-centric):
        row 0: dummy "image" placeholder      -> [0, 0, 0, 0]
        row 1: agent                          -> [ax, ay, dir, 1]
        row 2: goal                           -> [gx, gy, 0, 1]
        row 3: wall (first wall found)        -> [wx, wy, 0, 1]
        row 4:enemy                           -> [ex, ey, dir, 1]

    Shapes:
        logic_state:  (4, 4)
        neural_state: (4, 84, 84)
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

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None, num_balls=None):
        super().__init__(mode)

        self.render_mode = render_mode
        self.seed = seed
        self.num_balls = num_balls

        env_kwargs = {}

        # Only set n_obstacles if user explicitly passed a number
        if self.num_balls is not None:
            env_kwargs["n_obstacles"] = self.num_balls

        self.env = gym.make(
            "MiniGrid-Dynamic-Obstacles-6x6-v0",
            render_mode=render_mode,
            **env_kwargs
        )

        self.env = FullyObsWrapper(self.env)

        # 4 objects x 4 features
        self.n_objects = 5
        self.n_features = 4

        self.n_actions = 7
        self.n_raw_actions = 7


    def reset(self):
        if self.seed is not None:
            obs, info = self.env.reset(seed=self.seed)
        else:
            obs, info = self.env.reset()

        img = th.tensor(obs["image"], dtype=th.float32)  # (5,5,3)

        logic_state = self.extract_logic_state_objects()
        neural_state = self.extract_neural_state(img)

        return logic_state.unsqueeze(0), neural_state.unsqueeze(0)

    def step(self, action, is_mapped: bool = False):
        uenv = self.env.unwrapped
        self.last_agent_pos = tuple(uenv.agent_pos)
        self.last_obstacle_positions = [tuple(obj.cur_pos) for obj in uenv.obstacles]

        obs, reward, terminated, truncated, info = self.env.step(int(action))
        done = terminated or truncated

        img = th.tensor(obs["image"], dtype=th.float32)

        logic_state = self.extract_logic_state_objects()
        neural_state = self.extract_neural_state(img)




        # lazy import so BlendRL doesn't load it for other envs
        import importlib

        if not hasattr(self, "reward_model"):
            module = importlib.import_module("in.envs.minigrid.blenderl_reward")
            MiniGridReward = module.MiniGridReward
            self.reward_model = MiniGridReward()

        # initialize distance after finishing an episode
        if done:
            self.reward_model.reset(logic_state.unsqueeze(0))

        # compute shaped reward
        shaped = self.reward_model.reward_function(
            logic_state.unsqueeze(0),  # batch format
            int(action),
            reward
        )

        # After stepping the env
        agent_pos = tuple(uenv.agent_pos)
        obstacle_positions = [tuple(obj.cur_pos) for obj in uenv.obstacles]

        enemy_collision = False

        # Case 1: obstacle moved into agent
        if agent_pos in self.last_obstacle_positions:
            enemy_collision = True

        # Case 2: agent moved into obstacle (rare)
        if agent_pos in obstacle_positions:
            enemy_collision = True

        # Return info
        if enemy_collision:
            info["enemy_collision"] = True


        # --- DEBUG: detect enemy collision at episode end ---
        if done:
            if info.get("dynamic_obstacle", False):
                print("[DEBUG] Episode ended due to collision with dynamic obstacle.")
            elif info.get("static_obstacle", False):
                print("[DEBUG] Episode ended due to collision with static obstacle.")
            elif reward > 0:
                print("[DEBUG] Episode ended because goal was reached.")
            else:
                print(f"[DEBUG] {info}" )

        return (
            (logic_state, neural_state),
            [shaped],
            [truncated],
            [done],
            [info],
        )

    def extract_logic_state_objects(self) -> th.Tensor:
        """
        row 0: [0,0,0,0]         dummy "image"
        row 1: [ax,ay,dir,1]     agent
        row 2: [gx,gy,0,1]       goal
        row 3: [wx,wy,0,1]       wall
        row 4: [ex,ey,0,1]       nearest enemy (Ball / dynamic obstacle)
        """
        env = self.env

        # access underlying env to avoid wrapper warnings
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

        # --- ENEMIES (Option B: summarize all via nearest enemy) ---
        enemy_positions = []

        # Primary source: dynamic obstacles from the env
        if hasattr(uenv, "obstacles") and uenv.obstacles is not None:
            enemy_positions.extend([tuple(obj.cur_pos) for obj in uenv.obstacles])

        # Fallback: scan grid for Ball objects (in case obstacles list isn't present)
        if not enemy_positions:
            for x in range(uenv.width):
                for y in range(uenv.height):
                    obj = uenv.grid.get(x, y)
                    if isinstance(obj, Ball):
                        enemy_positions.append((x, y))

        # Summarize enemies as the NEAREST one to the agent
        if enemy_positions:
            # Manhattan distance
            dists = [
                (abs(ax - ex) + abs(ay - ey), ex, ey)
                for (ex, ey) in enemy_positions
            ]
            dists.sort()
            nearest_dist, ex, ey = dists[0]
            # we keep col2=0, col3=1 so NSFR encoding stays compatible
            enemy_row = [ex, ey, 0, 1]
        else:
            # No enemies visible; use sentinel
            enemy_row = [-1, -1, 0, 1]

        logic = th.tensor(
            [
                [0, 0, 0, 0],
                [ax, ay, ad, 1],
                [gx, gy, 0, 1],
                [wx, wy, 0, 1],
                enemy_row,    # enemy
            ],
            dtype=th.int32,
        )
        return logic

    def extract_neural_state(self, img: th.Tensor) -> th.Tensor:
        #assert img.numel() == 75, f"Expected 75 elements (5x5x3), got {img.numel()}"

        x = img.permute(2, 0, 1).unsqueeze(0)  # (1,3,5,5)
        gray = x.mean(dim=1, keepdim=True)     # (1,1,5,5)
        gray_84 = F.interpolate(gray, size=(84, 84), mode="nearest")  # (1,1,84,84)
        stacked = gray_84.repeat(1, 4, 1, 1)   # (1,4,84,84)
        return stacked.squeeze(0)              # (4,84,84)

    def close(self):
        self.env.close()
