import torch as th
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
import numpy as np
import os

class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    name = "mimic"
    pred2action = {
        "withhold": 0,
        "administer": 1,
    }
    
    def __init__(
        self,
        mode: str,
        n_envs: int,
        seed=None,
        dataset_name="mimic_lazy_0_interventions_balanced.npz",
        **kwargs
    ):
        super().__init__(mode)
        self.n_envs = n_envs
        self.seed = seed if seed is not None else 42
        
        # Load dataset
        mimic_dir = "/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2"
        path = os.path.join(mimic_dir, dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"MIMIC dataset not found at {path}")
            
        data = np.load(path, allow_pickle=True)
        self.X = data['X']  # (N, 240, 49)
        self.y = data['y']  # (N, 1)
        self.mask = data['mask']  # (N, 240, 1)
        
        self.n_patients = len(self.X)
        self.states = self.X[:, :, :46]
        self.actions_antibiotics = self.X[:, :, 47]  # Column 47 is AntiInfectiveAdmin_max
        
        self.n_actions = 2
        self.n_raw_actions = 2
        self.n_features = 46
        
        # Keep track of active trajectories and steps for each vectorized env
        self.rng = np.random.default_rng(self.seed)
        self.current_traj_idx = np.zeros(n_envs, dtype=np.int32)
        self.current_step_idx = np.zeros(n_envs, dtype=np.int32)
        
        # Store valid steps for each patient stay
        self.valid_steps_by_patient = {}
        for i in range(self.n_patients):
            valid = np.where(self.mask[i].squeeze() != -1)[0]
            self.valid_steps_by_patient[i] = valid
            
        # Initial trajectory selection
        for i in range(n_envs):
            self._reset_env_slot(i)

    def _reset_env_slot(self, env_idx: int):
        """Randomly select a patient trajectory and set step to 0"""
        while True:
            idx = self.rng.integers(0, self.n_patients)
            if len(self.valid_steps_by_patient[idx]) > 0:
                break
        self.current_traj_idx[env_idx] = idx
        self.current_step_idx[env_idx] = 0

    def reset(self):
        logic_states = []
        neural_states = []
        
        for i in range(self.n_envs):
            self._reset_env_slot(i)
            traj = self.current_traj_idx[i]
            step_t = self.valid_steps_by_patient[traj][0]
            obs = self.states[traj, step_t]
            
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
            valid_steps = self.valid_steps_by_patient[traj]
            
            t = valid_steps[step_idx]
            
            # Action taken by policy
            policy_action = actions[i]
            if hasattr(policy_action, "item"):
                policy_action = policy_action.item()
                
            # Historical action taken by clinician
            clinician_action = int(self.actions_antibiotics[traj, t])
            
            # Outcome-weighted behavioral reward
            # episode length T
            T = len(valid_steps)
            outcome = self.y[traj, 0]
            
            if outcome == 0:  # Patient survived
                # Positive reinforcement for copying clinician
                reward = 1.0 / T if policy_action == clinician_action else 0.0
            else:  # Patient died
                # Negative reinforcement for copying clinician (force divergence)
                reward = -1.0 / T if policy_action == clinician_action else 0.0
            
            is_done = (step_idx == len(valid_steps) - 1)
            
            if is_done:
                terminated = True
                self._reset_env_slot(i)
                new_traj = self.current_traj_idx[i]
                new_t = self.valid_steps_by_patient[new_traj][0]
                next_obs = self.states[new_traj, new_t]
            else:
                terminated = False
                self.current_step_idx[i] += 1
                next_t = valid_steps[self.current_step_idx[i]]
                next_obs = self.states[traj, next_t]
                
            logic_state, neural_state = self.extract_logic_state(next_obs), self.extract_neural_state(next_obs)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(False)
            infos.append({})
            
        return (th.stack(logic_states), th.stack(neural_states)), np.array(rewards, dtype=np.float32), np.array(terminations, dtype=bool), np.array(truncations, dtype=bool), infos

    def extract_logic_state(self, obs):
        return th.tensor(obs, dtype=th.float32)

    def extract_neural_state(self, obs):
        return th.tensor(obs, dtype=th.float32)

    def get_action_meanings(self):
        return ["withhold", "administer"]

    def close(self):
        pass
