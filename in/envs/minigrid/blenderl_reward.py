class MiniGridReward:
    def __init__(self):
        self.prev_dist = None

    def reset(self, logic_state):
        agent = logic_state[0,1]
        goal  = logic_state[0,2]

        ax, ay = agent[0].item(), agent[1].item()
        gx, gy = goal [0].item(), goal [1].item()

        self.prev_dist = abs(ax - gx) + abs(ay - gy)

    def reward_function(self, logic_state, action, raw_env_reward):
        logic = logic_state[0]
        agent = logic[1]
        goal  = logic[2]

        ax, ay = agent[0].item(), agent[1].item()
        gx, gy = goal [0].item(), goal [1].item()

        curr_dist = abs(ax - gx) + abs(ay - gy)

        reward = 0.0

        # --- Base goal reward ---
        if raw_env_reward > 0:
            # Big success reward
            #larg reward when reaching goal since its the ultimate objective
            reward += 10.0

        # --- Distance shaping ---
        if self.prev_dist is not None:
            if curr_dist < self.prev_dist:
                reward += 0.20      # get rewarded if you move closer to the goal
            elif curr_dist > self.prev_dist:
                reward -= 0.10      # get penalized if you move away

        # --- Forward bad step penalty ---
        #punished if you run into a wall
        if action == 2:  # move_forward
            if curr_dist == self.prev_dist:
                reward -= 0.10      # softer than before

        # --- Turn penalty (only if not improving orientation) ---
        if action in (3,4):  # turn_left or turn_right
            reward -= 0.002   # tiny penalty

        # --- Small living penalty ---
        reward -= 0.001

        # update memory
        self.prev_dist = curr_dist

        return reward
