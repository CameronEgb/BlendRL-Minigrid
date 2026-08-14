import os
import pickle
import torch
import numpy as np
from pathlib import Path

class DatasetWriter:
    def __init__(self, save_dir, chunk_size=100000, env_name="env"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_idx = 0
        self.env_name = env_name
        self.total_steps = 0

    def add(self, obs, logic_obs, action, reward, next_obs, next_logic_obs, done):
        """
        Add a transition.
        obs: tensor or array
        logic_obs: tensor or array (can be None)
        action: tensor or array
        reward: float or tensor
        next_obs: tensor or array
        next_logic_obs: tensor or array (can be None)
        done: bool or tensor
        """
        def to_cpu(x, is_obs=False):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if is_obs and x is not None:
                # Use float32 for vector environments, uint8 only for images (> 2D)
                if len(x.shape) > 2:
                    return x.astype(np.uint8)
                return x.astype(np.float32)
            return x

        transition = {
            "obs": to_cpu(obs, is_obs=True),
            "logic_obs": to_cpu(logic_obs) if logic_obs is not None else None,
            "action": to_cpu(action),
            "reward": to_cpu(reward),
            "next_obs": to_cpu(next_obs, is_obs=True),
            "next_logic_obs": to_cpu(next_logic_obs) if next_logic_obs is not None else None,
            "done": to_cpu(done)
        }
        self.buffer.append(transition)
        
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def batch_add(self, obs, logic_obs, action, reward, next_obs, next_logic_obs, done):
        """
        Add a batch of transitions.
        """
        # Ensure input is batch-like (at least 1D)
        if len(obs.shape) == 1: # Single vector obs
             batch_size = 1
             obs = obs.unsqueeze(0)
             if logic_obs is not None: logic_obs = logic_obs.unsqueeze(0)
             action = action.unsqueeze(0)
             reward = torch.tensor([reward]) if not isinstance(reward, torch.Tensor) else reward.unsqueeze(0)
             next_obs = next_obs.unsqueeze(0)
             if next_logic_obs is not None: next_logic_obs = next_logic_obs.unsqueeze(0)
             done = torch.tensor([done]) if not isinstance(done, torch.Tensor) else done.unsqueeze(0)
        else:
            batch_size = len(obs)
        
        def to_cpu(x, is_obs=False):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if is_obs and x is not None:
                # Use float32 for vector environments like CartPole/MountainCar
                # Use uint8 only for images
                if len(x.shape) > 2: # Image-like (H, W, C)
                    return x.astype(np.uint8)
                return x.astype(np.float32)
            return x

        obs_cpu = to_cpu(obs, is_obs=True)
        logic_obs_cpu = to_cpu(logic_obs) if logic_obs is not None else None
        action_cpu = to_cpu(action)
        reward_cpu = to_cpu(reward)
        next_obs_cpu = to_cpu(next_obs, is_obs=True)
        next_logic_obs_cpu = to_cpu(next_logic_obs) if next_logic_obs is not None else None
        done_cpu = to_cpu(done)

        for i in range(batch_size):
            transition = {
                "obs": obs_cpu[i],
                "logic_obs": logic_obs_cpu[i] if logic_obs_cpu is not None else None,
                "action": action_cpu[i],
                "reward": reward_cpu[i],
                "next_obs": next_obs_cpu[i],
                "next_logic_obs": next_logic_obs_cpu[i] if next_logic_obs_cpu is not None else None,
                "done": done_cpu[i]
            }
            self.buffer.append(transition)
        
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        
        filename = self.save_dir / f"dataset_{self.env_name}_{self.chunk_idx:05d}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)
        
        # print(f"Saved dataset chunk {self.chunk_idx} with {len(self.buffer)} transitions to {filename}")
        self.total_steps += len(self.buffer)
        self.buffer = []
        self.chunk_idx += 1

    def close(self):
        self.flush()

class DatasetReader:
    def __init__(self, dataset_dirs, device="cpu"):
        self.device = device
        self.files = []
        if isinstance(dataset_dirs, (str, Path)):
            dataset_dirs = [dataset_dirs]
            
        for d in dataset_dirs:
            p = Path(d)
            if p.exists():
                self.files.extend(sorted(list(p.glob("*.pkl"))))
        
        # Self-healing: if no PKL chunks found, check if a matching NPZ file exists to auto-convert
        if not self.files:
            import subprocess
            for d in dataset_dirs:
                p = Path(d)
                npz_candidate = p.with_suffix(".npz")
                if not npz_candidate.exists():
                    npz_candidate = p.parent / f"{p.name}.npz"
                if npz_candidate.exists():
                    print(f"\n[DatasetReader] Detected NPZ dataset '{npz_candidate}' with no PKL chunks.")
                    print(f"[DatasetReader] Auto-converting NPZ to PKL format at '{p}'...")
                    script_path = Path(__file__).resolve().parent.parent / "scripts" / "convert_npz_to_pkl.py"
                    subprocess.run([sys.executable, str(script_path), str(npz_candidate)], check=True)
                    if p.exists():
                        self.files.extend(sorted(list(p.glob("*.pkl"))))
                    break

        if not self.files:
            print(f"Warning: No dataset files found in {dataset_dirs}")

        import numpy as np
        import pickle
        import torch

        obs_list, logic_obs_list, actions_list = [], [], []
        rewards_list, next_obs_list, next_logic_obs_list, dones_list = [], [], [], []
        has_logic = False
        
        for f in self.files:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
                if not data: continue
                if not has_logic and data[0].get("logic_obs") is not None:
                    has_logic = True
                
                obs_list.append(np.array([t["obs"] for t in data]))
                if has_logic: logic_obs_list.append(np.array([t["logic_obs"] for t in data]))
                actions_list.append(np.array([t["action"] for t in data]))
                rewards_list.append(np.array([t["reward"] for t in data]))
                next_obs_list.append(np.array([t["next_obs"] for t in data]))
                if has_logic: next_logic_obs_list.append(np.array([t["next_logic_obs"] for t in data]))
                dones_list.append(np.array([t["done"] for t in data]))

        if obs_list:
            # Do not force float32 here! Preserve uint8 for memory efficiency on image environments
            self.obs = torch.tensor(np.concatenate(obs_list, axis=0))
            self.actions = torch.tensor(np.concatenate(actions_list, axis=0))
            self.rewards = torch.tensor(np.concatenate(rewards_list, axis=0))
            self.next_obs = torch.tensor(np.concatenate(next_obs_list, axis=0))
            self.dones = torch.tensor(np.concatenate(dones_list, axis=0))
            
            from src.pipeline.env_hooks import load_env_hooks
            env_name = os.environ.get("BLENDRL_ENV_NAME", "")
            hooks = load_env_hooks(env_name)
            hooks.transform_rewards(self, None)  # cfg not available here, use env vars
            
            if has_logic:
                self.logic_obs = torch.tensor(np.concatenate(logic_obs_list, axis=0))
                self.next_logic_obs = torch.tensor(np.concatenate(next_logic_obs_list, axis=0))
            else:
                self.logic_obs = None
                self.next_logic_obs = None
        else:
            self.obs = torch.tensor([])
            
        self.limit = len(self.obs)
    
    @property
    def device(self):
        return getattr(self, "_device", "cpu")

    @device.setter
    def device(self, dev):
        self._device = dev
        # For vector/tabular datasets (non-image, obs.ndim <= 2), preload directly onto GPU VRAM
        if hasattr(self, "obs") and isinstance(self.obs, torch.Tensor) and self.obs.numel() > 0 and self.obs.ndim <= 2 and str(dev) != "cpu":
            try:
                self.obs = self.obs.to(dev, dtype=torch.float32)
                self.actions = self.actions.to(dev, dtype=torch.long)
                self.rewards = self.rewards.to(dev, dtype=torch.float32)
                self.next_obs = self.next_obs.to(dev, dtype=torch.float32)
                self.dones = self.dones.to(dev, dtype=torch.float32)
                if self.logic_obs is not None:
                    self.logic_obs = self.logic_obs.to(dev, dtype=torch.float32)
                if self.next_logic_obs is not None:
                    self.next_logic_obs = self.next_logic_obs.to(dev, dtype=torch.float32)
            except Exception as e:
                print(f"Notice: Could not preload dataset to device {dev}: {e}")

    def to(self, device):
        self.device = device
        return self

    def set_limit(self, limit):
        new_limit = min(limit, len(self.obs))
        if new_limit != self.limit:
            self.limit = new_limit
            print(f"Dataset limit set to {self.limit} transitions.")

    def sample(self, batch_size, last=False):
        target_device = self.obs.device if (isinstance(self.obs, torch.Tensor) and self.obs.is_cuda) else "cpu"
        if last:
            start = max(0, self.limit - batch_size)
            idxs = torch.arange(start, self.limit, device=target_device)
            # If batch_size > available transitions, we just take what we have
        else:
            idxs = torch.randint(0, self.limit, (batch_size,), device=target_device)
        
        # Cast to correct types on the fly during transfer to device
        batch = {
            "obs": self.obs[idxs].to(self.device, dtype=torch.float32),
            "action": self.actions[idxs].to(self.device, dtype=torch.long),
            "reward": self.rewards[idxs].to(self.device, dtype=torch.float32),
            "next_obs": self.next_obs[idxs].to(self.device, dtype=torch.float32),
            "done": self.dones[idxs].to(self.device, dtype=torch.float32)
        }
        
        if self.logic_obs is not None:
            batch["logic_obs"] = self.logic_obs[idxs].to(self.device, dtype=torch.float32)
            batch["next_logic_obs"] = self.next_logic_obs[idxs].to(self.device, dtype=torch.float32)
        else:
            batch["logic_obs"] = None
            batch["next_logic_obs"] = None
            
        return batch

    def get_batch(self, idxs, device=None):
        if device is None:
            device = self.device
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs, dtype=torch.long)
        elif not isinstance(idxs, torch.Tensor):
            idxs = torch.tensor(idxs, dtype=torch.long)
        
        target_device = self.obs.device if (isinstance(self.obs, torch.Tensor) and self.obs.is_cuda) else "cpu"
        idxs = idxs.to(target_device)
        
        batch = {
            "obs": self.obs[idxs].to(device, dtype=torch.float32),
            "action": self.actions[idxs].to(device, dtype=torch.long),
            "reward": self.rewards[idxs].to(device, dtype=torch.float32),
            "next_obs": self.next_obs[idxs].to(device, dtype=torch.float32),
            "done": self.dones[idxs].to(device, dtype=torch.float32)
        }
        
        if self.logic_obs is not None:
            batch["logic_obs"] = self.logic_obs[idxs].to(device, dtype=torch.float32)
            batch["next_logic_obs"] = self.next_logic_obs[idxs].to(device, dtype=torch.float32)
        else:
            batch["logic_obs"] = None
            batch["next_logic_obs"] = None
            
        return batch

    def split(self, val_ratio=0.1, seed=42):
        """Split this reader into (train_reader, val_reader) by trajectory where possible."""
        n_total = len(self.obs)
        if n_total <= 1 or val_ratio <= 0.0 or val_ratio >= 1.0:
            return self, None
            
        rng = np.random.default_rng(seed)
        
        # If dones has trajectory markers, split by complete trajectories
        done_indices = torch.where(self.dones == 1.0)[0].cpu().numpy()
        if len(done_indices) > 1:
            trajectories = []
            start = 0
            for end_idx in done_indices:
                trajectories.append(list(range(start, int(end_idx) + 1)))
                start = int(end_idx) + 1
            if start < n_total:
                trajectories.append(list(range(start, n_total)))
                
            n_val_trajs = max(1, int(round(len(trajectories) * val_ratio)))
            shuffled_traj_indices = rng.permutation(len(trajectories))
            val_traj_indices = set(shuffled_traj_indices[:n_val_trajs])
            
            train_indices = []
            val_indices = []
            for t_idx, traj in enumerate(trajectories):
                if t_idx in val_traj_indices:
                    val_indices.extend(traj)
                else:
                    train_indices.extend(traj)
        else:
            # Fallback to random transition split
            shuffled = rng.permutation(n_total)
            n_val = max(1, int(round(n_total * val_ratio)))
            val_indices = shuffled[:n_val].tolist()
            train_indices = shuffled[n_val:].tolist()
            
        train_reader = DatasetReader.__new__(DatasetReader)
        train_reader._device = "cpu"
        train_reader.files = self.files
        train_reader.obs = self.obs[train_indices]
        train_reader.actions = self.actions[train_indices]
        train_reader.rewards = self.rewards[train_indices]
        train_reader.next_obs = self.next_obs[train_indices]
        train_reader.dones = self.dones[train_indices]
        train_reader.logic_obs = self.logic_obs[train_indices] if self.logic_obs is not None else None
        train_reader.next_logic_obs = self.next_logic_obs[train_indices] if self.next_logic_obs is not None else None
        train_reader.limit = len(train_reader.obs)
        
        val_reader = DatasetReader.__new__(DatasetReader)
        val_reader._device = "cpu"
        val_reader.files = self.files
        val_reader.obs = self.obs[val_indices]
        val_reader.actions = self.actions[val_indices]
        val_reader.rewards = self.rewards[val_indices]
        val_reader.next_obs = self.next_obs[val_indices]
        val_reader.dones = self.dones[val_indices]
        val_reader.logic_obs = self.logic_obs[val_indices] if self.logic_obs is not None else None
        val_reader.next_logic_obs = self.next_logic_obs[val_indices] if self.next_logic_obs is not None else None
        val_reader.limit = len(val_reader.obs)
        
        print(f"[Dataset Split] Train set: {len(train_reader)} transitions | Validation set: {len(val_reader)} transitions (val_ratio={val_ratio})")
        return train_reader, val_reader

    def __len__(self):
        return len(self.obs)

