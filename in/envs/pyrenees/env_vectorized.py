import torch as th
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
import numpy as np
import os

class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    name = "pyrenees"
    pred2action = {
        "action_0": 0,
        "action_1": 1,
    }
    
    def __init__(
        self,
        mode: str,
        n_envs: int,
        seed=None,
        dataset_name=None,
        **kwargs
    ):
        super().__init__(mode)
        if dataset_name is None:
            dataset_name = "pyrenees_clean.npz"
        self.n_envs = n_envs
        self.seed = seed if seed is not None else 42
        
        # Locate Pyrenees dataset .npz file
        pyrenees_dir = os.environ.get("PYRENEES_DATASET_DIR", "")
        if not pyrenees_dir or not os.path.exists(pyrenees_dir):
            pyrenees_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/pyrenees"))
        if not pyrenees_dir or not os.path.exists(pyrenees_dir):
            pyrenees_dir = os.path.abspath(os.path.join(os.getcwd(), "in/datasets/pyrenees"))
            
        path = os.path.join(pyrenees_dir, dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pyrenees dataset npz not found at {path}. Please run preprocess_pyrenees.py first.")
            
        data = np.load(path, allow_pickle=True)
        self.states = data['states']        # Array of object or (N, T, 123)
        self.actions = data['actions']      # Array of object or (N, T)
        self.rewards = data['rewards']      # Array of object or (N, T)
        self.dones = data['dones']          # Array of object or (N, T)
        
        self.n_trajectories = len(self.states)
        self.n_actions = 2
        self.n_raw_actions = 2
        self.n_features = 123
        
        # RNG and vector env pointers
        self.rng = np.random.default_rng(self.seed)
        self.current_traj_idx = np.zeros(n_envs, dtype=np.int32)
        self.current_step_idx = np.zeros(n_envs, dtype=np.int32)
        
        for i in range(n_envs):
            self._reset_env_slot(i)

    def _reset_env_slot(self, env_idx: int):
        """Select a random student trajectory and set step index to 0."""
        traj_idx = self.rng.integers(0, self.n_trajectories)
        self.current_traj_idx[env_idx] = traj_idx
        self.current_step_idx[env_idx] = 0

    def reset(self):
        logic_states = []
        neural_states = []
        
        for i in range(self.n_envs):
            self._reset_env_slot(i)
            traj = self.current_traj_idx[i]
            step_t = self.current_step_idx[i]
            obs = self.states[traj][step_t]
            
            logic_state, neural_state = self.extract_logic_state(obs), self.extract_neural_state(obs)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            
        return th.stack(logic_states), th.stack(neural_states)

    def step(self, actions, is_mapped: bool = False):
        rewards = []
        terminations = []
        truncations = []
        infos = []
        logic_states = []
        neural_states = []
        
        for i in range(self.n_envs):
            traj = self.current_traj_idx[i]
            step_idx = self.current_step_idx[i]
            traj_len = len(self.states[traj])
            
            # Record reward & action matching
            reward = float(self.rewards[traj][step_idx])
            is_done = (step_idx >= traj_len - 1) or bool(self.dones[traj][step_idx])
            
            if is_done:
                terminated = True
                self._reset_env_slot(i)
                new_traj = self.current_traj_idx[i]
                next_obs = self.states[new_traj][0]
            else:
                terminated = False
                self.current_step_idx[i] += 1
                next_obs = self.states[traj][self.current_step_idx[i]]
                
            logic_state, neural_state = self.extract_logic_state(next_obs), self.extract_neural_state(next_obs)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(False)
            infos.append({})
            
        return (th.stack(logic_states), th.stack(neural_states)), np.array(rewards, dtype=np.float32), np.array(terminations, dtype=bool), np.array(truncations, dtype=bool), infos

    def extract_logic_state(self, obs):
        state = th.zeros((2, 123), dtype=th.float32)
        state[0] = th.tensor(obs, dtype=th.float32)
        state[1] = th.tensor(obs, dtype=th.float32)
        return state

    def extract_neural_state(self, obs):
        return th.tensor(obs, dtype=th.float32)

    def get_action_meanings(self):
        return ["action_0", "action_1"]

    def close(self):
        pass
