"""
Pyrenees VectorizedNudgeEnv — Dynamic multi-problem BlendRL environment for the Pyrenees ITS dataset.

Supports both:
  1. Problem-Level Policy (problem.csv):
     - Action space (3 actions): 0 = PS, 1 = WE, 2 = FWE
     - State representation: 130 features + 3 alternation slots = 133 dimensions
  2. Step-Level Policy (ex132(w).csv, etc.):
     - Action space (2 actions): 0 = PS (Elicit), 1 = WE (Tell)
     - State representation: 123 features + 3 alternation slots = 126 dimensions
"""

import os
from pathlib import Path
import numpy as np
import torch as th
from blendrl.env_vectorized import VectorizedNudgeBaseEnv


class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    name = "pyrenees"

    def __init__(
        self,
        mode: str,
        n_envs: int,
        seed=None,
        dataset_name=None,
        problem_type=None,
        **kwargs,
    ):
        super().__init__(mode)
        self.n_envs = n_envs
        self.seed = seed if seed is not None else 42

        # ── Determine problem type ─────────────────────────────────────────
        if problem_type is None:
            problem_type = os.environ.get("PYRENEES_PROBLEM_TYPE", None)

        if problem_type is None and dataset_name is not None:
            if "per_problem" in dataset_name or ".csv" in dataset_name:
                p = Path(dataset_name)
                problem_type = p.stem.replace(".npz", "")
                if problem_type == "clean":
                    problem_type = p.parent.name
            elif dataset_name in ["problem", "problem.npz"]:
                problem_type = "problem"

        if problem_type is None:
            problem_type = "problem"

        self.problem_type = problem_type
        self.is_problem_level = (self.problem_type == "problem")

        # ── Configure action and state dimensions ──────────────────────────
        if self.is_problem_level:
            self.n_actions = 3
            self.n_raw_actions = 3
            self.n_features = 130
            self.pred2action = {
                "action_ps": 0,   # Problem Solving
                "action_we": 1,   # Worked Example
                "action_fwe": 2,  # Faded Worked Example
            }
        else:
            self.n_actions = 2
            self.n_raw_actions = 2
            self.n_features = 123
            self.pred2action = {
                "action_ps": 0,  # Elicit
                "action_we": 1,  # Tell
            }

        # Augmented dim = raw features + 3 alternation slots
        self.aug_dim = self.n_features + 3
        self.slot_ps = self.n_features
        self.slot_we = self.n_features + 1
        self.slot_fwe = self.n_features + 2
        self.action_to_slot = {0: self.slot_ps, 1: self.slot_we, 2: self.slot_fwe}

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

        candidate_paths = [
            os.path.join(pyrenees_dir, "per_problem", self.problem_type, "clean.npz"),
            os.path.join(pyrenees_dir, self.problem_type, "clean.npz"),
            os.path.join(pyrenees_dir, f"{self.problem_type}.npz"),
            os.path.join(pyrenees_dir, dataset_name if dataset_name else "pyrenees_clean.npz"),
            os.path.join(pyrenees_dir, "pyrenees_clean.npz"),
        ]

        found_path = None
        for cand in candidate_paths:
            if cand and os.path.exists(cand):
                found_path = cand
                break

        if found_path is None:
            raise FileNotFoundError(
                f"Pyrenees dataset npz for problem '{self.problem_type}' not found. "
                f"Searched: {candidate_paths}. Please run preprocess_pyrenees_per_problem.py first."
            )

        data = np.load(found_path, allow_pickle=True)
        self.states = data["states"]    # (N,) array of (T, n_features) arrays
        self.actions = data["actions"]  # (N,) array of (T,) arrays
        self.rewards = data["rewards"]  # (N,) array of (T,) arrays
        self.dones = data["dones"]      # (N,) array of (T,) arrays

        # Validate feature dimension of loaded data
        if len(self.states) > 0 and len(self.states[0]) > 0:
            actual_feats = self.states[0].shape[-1]
            self.n_features = actual_feats
            self.aug_dim = self.n_features + 3
            self.slot_ps = self.n_features
            self.slot_we = self.n_features + 1
            self.slot_fwe = self.n_features + 2
            self.action_to_slot = {0: self.slot_ps, 1: self.slot_we, 2: self.slot_fwe}

        self.n_trajectories = len(self.states)

        # ── RNG + per-env pointers ────────────────────────────────────────
        self.rng = np.random.default_rng(self.seed)
        self.current_traj_idx = np.zeros(n_envs, dtype=np.int32)
        self.current_step_idx = np.zeros(n_envs, dtype=np.int32)
        self._last_action = np.full(n_envs, fill_value=-1, dtype=np.int32)

        for i in range(n_envs):
            self._reset_env_slot(i)

    def _reset_env_slot(self, env_idx: int):
        traj_idx = self.rng.integers(0, self.n_trajectories)
        self.current_traj_idx[env_idx] = traj_idx
        self.current_step_idx[env_idx] = 0
        self._last_action[env_idx] = -1

    def _augment(self, raw_obs: np.ndarray, last_action: int) -> np.ndarray:
        aug = np.zeros(self.aug_dim, dtype=np.float32)
        aug[:self.n_features] = raw_obs[:self.n_features]
        if last_action in self.action_to_slot:
            aug[self.action_to_slot[last_action]] = 1.0
        return aug

    def reset(self):
        logic_states = []
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

            reward = float(self.rewards[traj][step_idx])
            is_done = (step_idx >= traj_len - 1) or bool(self.dones[traj][step_idx])

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
            np.array(rewards, dtype=np.float32),
            np.array(terminations, dtype=bool),
            np.array(truncations, dtype=bool),
            infos,
        )

    def extract_logic_state(self, obs_aug: np.ndarray) -> th.Tensor:
        state = th.zeros((2, self.aug_dim), dtype=th.float32)
        t = th.tensor(obs_aug, dtype=th.float32)
        state[0] = t  # student
        state[1] = t  # env
        return state

    def extract_neural_state(self, obs_aug: np.ndarray) -> th.Tensor:
        return th.tensor(obs_aug, dtype=th.float32)

    def get_action_meanings(self):
        if self.is_problem_level:
            return ["action_ps", "action_we", "action_fwe"]
        return ["action_ps", "action_we"]

    def close(self):
        pass
