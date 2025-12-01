import torch as th
import torch.nn.functional as F
import gymnasium as gym

from nudge.env import NudgeBaseEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.world_object import Goal, Wall, Ball


class NudgeEnv(NudgeBaseEnv):
    """
    MiniGrid NudgeEnv wrapper for BlendRL.


    Logic state (object-centric):
        row 0: dummy "image" placeholder      -> [0, 0, 0, 0]
        row 1: agent                          -> [ax, ay, dir, 1]
        row 2: goal                           -> [gx, gy, 0, 1]
        row 3: wall (first wall found)        -> [wx, wy, 0, 1]
        row 4:enemy                           -> [ex, ey, dir, 1]


    """
    name = "minigrid"

    pred2action = {
        "move_up": 0,
        "move_down": 1,
        "move_left": 2,
        "move_right": 3
    }

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None, num_balls=None):
        super().__init__(mode)

        self.render_mode = render_mode
        self.seed = seed
        self.num_balls = num_balls

        env_kwargs = {}

        # Only set n_obstacles if user explicitly passed a number, otherwise default for env is 3
        if self.num_balls is not None:
            env_kwargs["n_obstacles"] = self.num_balls

        #can change MiniGrid-Dynamic-Obstacles-6x6-v0 to another mingrid preset, have to change preds to match env
        self.env = gym.make(
            "MiniGrid-Dynamic-Obstacles-8x8-v0",
            render_mode=render_mode,
            **env_kwargs
        )

        self.env = FullyObsWrapper(self.env)

        self.max_obstacles = 5
        # 1 (dummy) + 1 (agent) + 1 (goal) + max_obstacles
        self.n_objects = 3 + self.max_obstacles
        self.n_features = 4

        self.n_actions = 4 # Updated to 4
        self.n_raw_actions = 4 # Updated to 4

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)

        img = th.tensor(obs["image"], dtype=th.float32)

        logic_state = self.extract_logic_state_objects()
        neural_state = self.extract_neural_state(img)

        return logic_state.unsqueeze(0), neural_state.unsqueeze(0)

    def step(self, action, is_mapped: bool = False):
        uenv = self.env.unwrapped
        self.last_agent_pos = tuple(uenv.agent_pos)
        self.last_obstacle_positions = [tuple(obj.cur_pos) for obj in uenv.obstacles]

        obs = None
        reward = None
        terminated = False
        truncated = False
        info = None

        uenv = self.env.unwrapped
        current_dir = uenv.agent_dir # 0: East, 1: South, 2: West, 3: North

        primitive_action_sequence = []

        # Determine target direction based on the custom action
        target_dir = -1
        if action == self.pred2action["move_up"]: # Move North
            target_dir = 3
        elif action == self.pred2action["move_down"]: # Move South
            target_dir = 1
        elif action == self.pred2action["move_left"]: # Move West
            target_dir = 2
        elif action == self.pred2action["move_right"]: # Move East
            target_dir = 0

        if target_dir != -1: # If it's one of the cardinal move actions
            # Calculate turns needed
            num_turns = (target_dir - current_dir + 4) % 4
            if num_turns == 3: # If 3 turns right, it's 1 turn left
                primitive_action_sequence.append(0) # Turn left
            else: # Otherwise, turn right num_turns times
                for _ in range(num_turns):
                    primitive_action_sequence.append(1) # Turn right
            primitive_action_sequence.append(2) # Move forward
        else: # If the action is not a custom cardinal move (shouldn't happen with updated pred2action)
            primitive_action_sequence.append(int(action)) # Execute the action as is

        done = False
        for p_action in primitive_action_sequence:
            obs, reward, terminated, truncated, info = self.env.step(p_action)
            done = terminated or truncated
            if done: # Stop if an episode ends during a sequence of primitive actions
                break

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
            shaped[0] -= 2.0


        # --- DEBUG: detect enemy collision at episode end ---
        """
        if done:
            if info.get("dynamic_obstacle", False):
                print("[DEBUG] Episode ended due to collision with dynamic obstacle.")
            elif info.get("static_obstacle", False):
                print("[DEBUG] Episode ended due to collision with static obstacle.")
            elif reward > 0:
                print("[DEBUG] Episode ended because goal was reached.")
            else:
                print(f"[DEBUG] {info}" )
        """

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
        row 3+: [ex,ey,0,1]      enemies (Ball / dynamic obstacle)
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
        
        logic_rows = [
            [0, 0, 0, 0],       # dummy
            [ax, ay, ad, 1],    # agent
            [gx, gy, 0, 1],     # goal
        ]

        # --- ENEMIES ---
        enemy_positions = []
        if hasattr(uenv, "obstacles") and uenv.obstacles is not None:
            enemy_positions.extend([tuple(obj.cur_pos) for obj in uenv.obstacles])

        # Add obstacles to logic state
        for i in range(self.max_obstacles):
            if i < len(enemy_positions):
                ex, ey = enemy_positions[i]
                logic_rows.append([ex, ey, 0, 1])
            else:
                # Pad with non-visible, out-of-bounds objects
                logic_rows.append([-1, -1, 0, 0])

        logic = th.tensor(logic_rows, dtype=th.int32)
        return logic

    def extract_neural_state(self, img: th.Tensor) -> th.Tensor:
        """
        Takes the symbolic grid representation and flattens it.
        """
        return img.float()

    def close(self):
        self.env.close()
