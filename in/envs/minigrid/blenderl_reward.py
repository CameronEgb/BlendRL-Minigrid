import torch as th

class MiniGridReward:
    """
    Reward shaping for MiniGrid tailored for BlendRL + NSFR.

    Uses:
    - +1.0 for reaching the goal
    - distance shaping
    - penalty for wall collisions
    - penalty for useless turning
    - very small living penalty
    """

    def __init__(self):
        self.prev_dist = None

    def reset(self, logic_state):
        # logic_state: (1,4,4)
        agent = logic_state[0,1]  # [ax, ay, dir, 1]
        goal  = logic_state[0,2]  # [gx, gy, 0, 1]

        ax, ay = agent[0].item(), agent[1].item()
        gx, gy = goal [0].item(), goal [1].item()

        self.prev_dist = abs(ax - gx) + abs(ay - gy)

    def reward_function(self, logic_state, action, raw_env_reward):
        # logic_state: (1,4,4)
        logic = logic_state[0]
        agent = logic[1]
        goal  = logic[2]

        ax, ay = agent[0].item(), agent[1].item()
        gx, gy = goal [0].item(), goal [1].item()

        curr_dist = abs(ax - gx) + abs(ay - gy)

        reward = 0.0

        # --- Base goal reward ---
        if raw_env_reward > 0:
            reward += 5.0

        # --- Distance shaping ---
        if self.prev_dist is not None:
            if curr_dist < self.prev_dist:
                reward += 0.05   # moved closer
            elif curr_dist > self.prev_dist:
                reward -= 0.02   # moved further

        # --- Wall-hit penalty ---
        if action == 2:  # move_forward
            if curr_dist == self.prev_dist:
                reward -= 0.50

        # --- Useless turning penalty ---
        if action in (3, 4):  # turn_left, turn_right
            reward -= 0.01

        # --- Living penalty to encourage finishing ---
        reward -= 0.005

        # Update distance tracker
        self.prev_dist = curr_dist

        return reward
