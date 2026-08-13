import os
import pickle
import torch
import numpy as np
from pathlib import Path
import lz4.frame # New import for LZ4 compression

class SeaquestDatasetWriter:
    def __init__(self, save_dir, chunk_size=100000, env_name="seaquest"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_idx = 0
        self.env_name = env_name
        self.total_steps = 0

    def batch_add(self, obs, logic_obs, action, reward, next_obs, next_logic_obs, done):
        """
        Add a batch of transitions.
        Optimized: Stores only the LATEST frame of next_obs to save space.
        Reconstruction happens in the Reader.
        Applies LZ4 compression and data type optimizations.
        """
        batch_size = len(obs)
        
        # Convert to numpy and optimize data types
        if isinstance(obs, torch.Tensor): obs = obs.detach().cpu().numpy()
        if isinstance(logic_obs, torch.Tensor): logic_obs = logic_obs.detach().cpu().numpy()
        if isinstance(action, torch.Tensor): action = action.detach().cpu().numpy()
        if isinstance(reward, torch.Tensor): reward = reward.detach().cpu().numpy()
        if isinstance(next_obs, torch.Tensor): next_obs = next_obs.detach().cpu().numpy()
        if isinstance(next_logic_obs, torch.Tensor): next_logic_obs = next_logic_obs.detach().cpu().numpy()
        if isinstance(done, torch.Tensor): done = done.detach().cpu().numpy()

        # Neural Obs (uint8 + LZ4 compression)
        # obs is (N, 4, 84, 84)
        obs_compressed = [lz4.frame.compress(o.astype(np.uint8).tobytes()) for o in obs]
        # next_obs is (N, 4, 84, 84), we only save the newest frame (N, 1, 84, 84)
        next_obs_latest_compressed = [lz4.frame.compress(n[-1:].astype(np.uint8).tobytes()) for n in next_obs]

        # Logic Obs (int32)
        logic_obs_processed = logic_obs.astype(np.int32)
        next_logic_obs_processed = next_logic_obs.astype(np.int32)

        # Action (uint8)
        action_processed = action.astype(np.uint8)

        # Reward (float32)
        reward_processed = reward.astype(np.float32)
        
        # Done (bool)
        done_processed = done.astype(np.bool_)

        for i in range(batch_size):
            transition = {
                "obs": obs_compressed[i],
                "logic_obs": logic_obs_processed[i],
                "action": action_processed[i],
                "reward": reward_processed[i],
                "next_obs_new": next_obs_latest_compressed[i],
                "next_logic_obs": next_logic_obs_processed[i],
                "done": done_processed[i]
            }
            self.buffer.append(transition)
        
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        # Store metadata for reconstruction and memmap creation
        metadata = {
            "num_transitions": len(self.buffer),
            "obs_shape": (4, 84, 84),
            "logic_obs_shape": self.buffer[0]["logic_obs"].shape if "logic_obs" in self.buffer[0] and self.buffer[0]["logic_obs"] is not None else None,
            "action_dtype": "uint8",
            "reward_dtype": "float32",
            "done_dtype": "bool_",
            "compression": "lz4"
        }
        
        filename = self.save_dir / f"dataset_{self.env_name}_{self.chunk_idx:05d}.pkl"
        with open(filename, "wb") as f:
            pickle.dump({"data": self.buffer, "metadata": metadata}, f)
        self.total_steps += len(self.buffer)
        self.buffer = []
        self.chunk_idx += 1

    def close(self):
        self.flush()

class SeaquestDatasetReader:
    def __init__(self, dataset_dirs, device="cpu"):
        self.device = device
        self.files = []
        if isinstance(dataset_dirs, (str, Path)):
            dataset_dirs = [dataset_dirs]
        for d in dataset_dirs:
            p = Path(d)
            if p.exists():
                self.files.extend(sorted(list(p.glob("*.pkl"))))
        
        if not self.files:
            print(f"Warning: No dataset files found in {dataset_dirs}")
            self.obs = np.array([])
            self.logic_obs = None
            self.actions = np.array([])
            self.rewards = np.array([])
            self.next_obs_new = np.array([])
            self.next_logic_obs = None
            self.dones = np.array([])
            self.limit = 0
            self.has_logic = False
            return

        # Phase 1: Collect metadata and total number of transitions
        total_transitions = 0
        obs_shape = (4, 84, 84) # Default, confirmed from env
        next_obs_new_shape = (1, 84, 84)
        logic_obs_shape = None
        action_dtype = np.uint8
        reward_dtype = np.float32
        done_dtype = np.bool_
        
        print(f"Scanning {len(self.files)} dataset chunks for metadata...")
        for f in self.files:
            with open(f, "rb") as fh:
                chunk_data = pickle.load(fh)
                metadata = chunk_data.get("metadata", {})
                total_transitions += metadata.get("num_transitions", len(chunk_data["data"]) if "data" in chunk_data else len(chunk_data))
                
                if metadata: # Use metadata from a chunk if available
                    obs_shape = metadata.get("obs_shape", obs_shape)
                    logic_obs_shape = metadata.get("logic_obs_shape", logic_obs_shape)
                    action_dtype = getattr(np, metadata.get("action_dtype", "uint8"))
                    reward_dtype = getattr(np, metadata.get("reward_dtype", "float32"))
                    done_dtype_str = metadata.get("done_dtype", "bool_")
                    if done_dtype_str == "bool":
                        done_dtype = np.bool_
                    else:
                        done_dtype = getattr(np, done_dtype_str)
        
        # Determine if we have logic data
        self.has_logic = logic_obs_shape is not None and len(logic_obs_shape) > 0
        
        # Create memory-mapped arrays
        base_path = Path(dataset_dirs[0]) if isinstance(dataset_dirs, list) else Path(dataset_dirs)
        self.memmap_dir = base_path / "memmap_cache"
        self.memmap_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating memory-mapped files in {self.memmap_dir} for {total_transitions} transitions...")

        self.obs = np.memmap(self.memmap_dir / "obs.mmap", dtype=np.uint8, mode='w+', shape=(total_transitions,) + obs_shape)
        self.next_obs_new = np.memmap(self.memmap_dir / "next_obs_new.mmap", dtype=np.uint8, mode='w+', shape=(total_transitions,) + next_obs_new_shape)
        self.actions = np.memmap(self.memmap_dir / "actions.mmap", dtype=action_dtype, mode='w+', shape=(total_transitions,))
        self.rewards = np.memmap(self.memmap_dir / "rewards.mmap", dtype=reward_dtype, mode='w+', shape=(total_transitions,))
        self.dones = np.memmap(self.memmap_dir / "dones.mmap", dtype=done_dtype, mode='w+', shape=(total_transitions,))
        
        if self.has_logic:
            self.logic_obs = np.memmap(self.memmap_dir / "logic_obs.mmap", dtype=np.int32, mode='w+', shape=(total_transitions,) + logic_obs_shape)
            self.next_logic_obs = np.memmap(self.memmap_dir / "next_logic_obs.mmap", dtype=np.int32, mode='w+', shape=(total_transitions,) + logic_obs_shape)
        else:
            self.logic_obs = None
            self.next_logic_obs = None

        # Phase 2: Load and decompress into memory-mapped arrays
        current_idx = 0
        print(f"Loading and decompressing {len(self.files)} chunks into memory-mapped files...")
        for f in self.files:
            with open(f, "rb") as fh:
                chunk_data = pickle.load(fh)
                metadata = chunk_data.get("metadata", {})
                transitions = chunk_data.get("data", [])
                
                num_transitions_in_chunk = len(transitions)
                end_idx = current_idx + num_transitions_in_chunk
                
                compression = metadata.get("compression", None)

                # Decompress and store
                for i, t in enumerate(transitions):
                    obs_data = t["obs"]
                    next_obs_data = t["next_obs_new"]
                    
                    if compression == "lz4":
                        self.obs[current_idx + i] = np.frombuffer(lz4.frame.decompress(obs_data), dtype=np.uint8).reshape(obs_shape)
                        self.next_obs_new[current_idx + i] = np.frombuffer(lz4.frame.decompress(next_obs_data), dtype=np.uint8).reshape(next_obs_new_shape)
                    else:
                        # Assume already decompressed / raw array or bytes
                        if isinstance(obs_data, bytes):
                            self.obs[current_idx + i] = np.frombuffer(obs_data, dtype=np.uint8).reshape(obs_shape)
                        else:
                            self.obs[current_idx + i] = obs_data
                        
                        if isinstance(next_obs_data, bytes):
                            self.next_obs_new[current_idx + i] = np.frombuffer(next_obs_data, dtype=np.uint8).reshape(next_obs_new_shape)
                        else:
                            self.next_obs_new[current_idx + i] = next_obs_data
                            
                    self.actions[current_idx + i] = t["action"]
                    self.rewards[current_idx + i] = t["reward"]
                    self.dones[current_idx + i] = t["done"]
                    if self.has_logic:
                        self.logic_obs[current_idx + i] = t["logic_obs"]
                        self.next_logic_obs[current_idx + i] = t["next_logic_obs"]
                current_idx = end_idx
                
        # Flush changes to disk
        self.obs.flush()
        self.next_obs_new.flush()
        self.actions.flush()
        self.rewards.flush()
        self.dones.flush()
        if self.has_logic:
            self.logic_obs.flush()
            self.next_logic_obs.flush()

        self.limit = total_transitions
        print(f"Dataset loaded: {self.limit} transitions (logic data: {'YES' if self.has_logic else 'NO'}).")

    def set_limit(self, limit):
        new_limit = min(limit, len(self.obs))
        if new_limit != self.limit:
            self.limit = new_limit
            print(f"Dataset limit set to {self.limit} transitions.")

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.limit, size=batch_size)
        
        obs_batch = torch.tensor(self.obs[idxs], device=self.device, dtype=torch.float32)
        next_new_frame = torch.tensor(self.next_obs_new[idxs], device=self.device, dtype=torch.float32)
        
        # TRICKY RECONSTRUCTION:
        # next_obs = [obs[1], obs[2], obs[3], next_new_frame]
        next_obs_batch = torch.cat([obs_batch[:, 1:, :, :], next_new_frame], dim=1)
        
        batch = {
            "obs": obs_batch,
            "action": torch.tensor(self.actions[idxs], device=self.device, dtype=torch.long),
            "reward": torch.tensor(self.rewards[idxs], device=self.device, dtype=torch.float32),
            "next_obs": next_obs_batch,
            "done": torch.tensor(self.dones[idxs], device=self.device, dtype=torch.float32),
        }
        
        if self.has_logic:
            batch["logic_obs"] = torch.tensor(self.logic_obs[idxs], device=self.device, dtype=torch.float32)
            batch["next_logic_obs"] = torch.tensor(self.next_logic_obs[idxs], device=self.device, dtype=torch.float32)
        else:
            batch["logic_obs"] = None
            batch["next_logic_obs"] = None
            
        return batch

    def __len__(self):
        return self.limit

    def cleanup_memmap_cache(self):
        """
        Removes the memory-mapped cache files. Call this when the dataset is no longer needed.
        """
        if hasattr(self, 'memmap_dir') and self.memmap_dir.exists():
            import shutil
            print(f"Cleaning up memory-mapped cache: {self.memmap_dir}")
            shutil.rmtree(self.memmap_dir)
