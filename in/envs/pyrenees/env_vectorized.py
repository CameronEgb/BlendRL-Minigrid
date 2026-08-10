"""
Pyrenees VectorizedNudgeEnv — BlendRL environment for the Pyrenees ITS dataset.

Action space (3 discrete actions):
    0 = PS  (Problem Solving)
    1 = WE  (Worked Example)
    2 = FWE (Faded Worked Example)

State representation (126-dim augmented vector):
    [0:123]  — z-scored performance features from pyrenees_clean.npz
    [123]    — last_was_ps  (1.0 if last assigned action was PS, else 0.0)
    [124]    — last_was_we  (1.0 if last assigned action was WE, else 0.0)
    [125]    — last_was_fwe (1.0 if last assigned action was FWE, else 0.0)

Logic state shape: (n_envs, 2, 126)  — two objects: [student, env]
Neural state shape: (n_envs, 126)
"""

import torch as th
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
import numpy as np
import os


# Feature indices for the augmented alternation slots
IDX_LAST_PS  = 123
IDX_LAST_WE  = 124
IDX_LAST_FWE = 125

# Action integer → augmented slot index
ACTION_TO_SLOT = {0: IDX_LAST_PS, 1: IDX_LAST_WE, 2: IDX_LAST_FWE}


class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    name = "pyrenees"

    # Predicate → raw action index mapping
    pred2action = {
        "action_ps":  0,  # Problem Solving
        "action_we":  1,  # Worked Example
        "action_fwe": 2,  # Faded Worked Example
    }

    def __init__(
        self,
        mode: str,
        n_envs: int,
        seed=None,
        dataset_name=None,
        **kwargs,
    ):
        super().__init__(mode)
        if dataset_name is None:
            dataset_name = "pyrenees_clean.npz"
        self.n_envs = n_envs
        self.seed = seed if seed is not None else 42

        # ── Locate dataset ────────────────────────────────────────────────
        pyrenees_dir = os.environ.get("PYRENEES_DATASET_DIR", "")
        if not pyrenees_dir or not os.path.exists(pyrenees_dir):
            pyrenees_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../datasets/pyrenees")
            )
        if not pyrenees_dir or not os.path.exists(pyrenees_dir):
            pyrenees_dir = os.path.abspath(
                os.path.join(os.getcwd(), "in/datasets/pyrenees")
            )

        path = os.path.join(pyrenees_dir, dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pyrenees dataset npz not found at {path}. "
                f"Please run preprocess_pyrenees.py first."
            )

        data = np.load(path, allow_pickle=True)
        self.states  = data["states"]   # (N,) of (T, 123) arrays
        self.actions = data["actions"]  # (N,) of (T,)   arrays
        self.rewards = data["rewards"]  # (N,) of (T,)   arrays
        self.dones   = data["dones"]    # (N,) of (T,)   arrays

        self.n_trajectories = len(self.states)
        self.n_actions      = 3   # PS, WE, FWE
        self.n_raw_actions  = 3
        self.n_features     = 123 # original feature dim

        # ── Augmented dim = 126 (123 features + 3 alternation slots) ─────
        self.aug_dim = 126

        # ── RNG + per-env pointers ────────────────────────────────────────
        self.rng = np.random.default_rng(self.seed)
        self.current_traj_idx = np.zeros(n_envs, dtype=np.int32)
        self.current_step_idx = np.zeros(n_envs, dtype=np.int32)

        # Tracks last action per env for alternation augmentation
        self._last_action = np.full(n_envs, fill_value=-1, dtype=np.int32)

        for i in range(n_envs):
            self._reset_env_slot(i)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _reset_env_slot(self, env_idx: int):
        """Pick a random trajectory and reset step pointer + alternation state."""
        traj_idx = self.rng.integers(0, self.n_trajectories)
        self.current_traj_idx[env_idx] = traj_idx
        self.current_step_idx[env_idx] = 0
        self._last_action[env_idx] = -1  # unknown / start-of-episode

    def _augment(self, obs_123: np.ndarray, last_action: int) -> np.ndarray:
        """Append 3 one-hot alternation bits to a 123-dim observation."""
        aug = np.zeros(self.aug_dim, dtype=np.float32)
        aug[:self.n_features] = obs_123
        if last_action in ACTION_TO_SLOT:
            aug[ACTION_TO_SLOT[last_action]] = 1.0
        return aug

    # ─── Public API ───────────────────────────────────────────────────────────

    def reset(self):
        logic_states  = []
        neural_states = []

        for i in range(self.n_envs):
            self._reset_env_slot(i)
            traj = self.current_traj_idx[i]
            step = self.current_step_idx[i]
            raw_obs = self.states[traj][step]
            aug_obs = self._augment(raw_obs, self._last_action[i])

            logic_states.append(self.extract_logic_state(aug_obs))
            neural_states.append(self.extract_neural_state(aug_obs))

        return th.stack(logic_states), th.stack(neural_states)

    def step(self, actions, is_mapped: bool = False):
        rewards      = []
        terminations = []
        truncations  = []
        infos        = []
        logic_states  = []
        neural_states = []

        for i in range(self.n_envs):
            traj     = self.current_traj_idx[i]
            step_idx = self.current_step_idx[i]
            traj_len = len(self.states[traj])

            reward  = float(self.rewards[traj][step_idx])
            is_done = (step_idx >= traj_len - 1) or bool(self.dones[traj][step_idx])

            # Record the action taken for alternation tracking
            self._last_action[i] = int(actions[i])

            if is_done:
                terminated = True
                self._reset_env_slot(i)
                new_traj = self.current_traj_idx[i]
                raw_next = self.states[new_traj][0]
            else:
                terminated = False
                self.current_step_idx[i] += 1
                raw_next = self.states[traj][self.current_step_idx[i]]

            aug_next = self._augment(raw_next, self._last_action[i])
            logic_states.append(self.extract_logic_state(aug_next))
            neural_states.append(self.extract_neural_state(aug_next))

            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(False)
            infos.append({})

        return (
            (th.stack(logic_states), th.stack(neural_states)),
            np.array(rewards,       dtype=np.float32),
            np.array(terminations,  dtype=bool),
            np.array(truncations,   dtype=bool),
            infos,
        )

    def extract_logic_state(self, obs_126: np.ndarray) -> th.Tensor:
        """
        Returns shape (2, 126): two object slots.
          obj1 = student state  (used by 'oagent' predicates)
          obj2 = env placeholder (used by 'oenv' predicates, e.g. true(E))
        """
        state = th.zeros((2, self.aug_dim), dtype=th.float32)
        t = th.tensor(obs_126, dtype=th.float32)
        state[0] = t  # student
        state[1] = t  # env (shared; true(E) only uses first dim anyway)
        return state

    def extract_neural_state(self, obs_126: np.ndarray) -> th.Tensor:
        return th.tensor(obs_126, dtype=th.float32)

    def get_action_meanings(self):
        return ["action_ps", "action_we", "action_fwe"]

    def close(self):
        pass
