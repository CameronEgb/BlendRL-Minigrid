import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from src.dataset_utils import DatasetReader
from typing import Dict, Any, Optional

class OfflineDataset(Dataset):
    def __init__(self, reader: DatasetReader):
        self.reader = reader
        
    def __len__(self):
        return self.reader.limit
        
    def __getitem__(self, idx):
        # We'll just return the index and let the agent sample to keep it efficient with the existing reader
        return idx

class RLDataModule(L.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.reader = None
        
    def setup(self, stage: Optional[str] = None):
        if self.cfg.mode.type == "offline":
            self.reader = DatasetReader(self.cfg.mode.dataset_path)
            self.dataset = OfflineDataset(self.reader)
        else:
            # Online mode dummy dataset
            self.dataset = torch.utils.data.TensorDataset(torch.zeros(1))
            
    def train_dataloader(self):
        # For online mode, cfg.agent.batch_size might not exist in all agent configs (e.g. standard PPO)
        # But for online rollouts, we only need a dummy dataloader that yields once per epoch.
        batch_size = self.cfg.agent.get("batch_size", 1)
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=1)
