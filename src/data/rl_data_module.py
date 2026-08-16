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
        return idx

class RLDataModule(L.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.reader = None
        self.val_reader = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        if self.cfg.mode.type == "offline":
            full_reader = DatasetReader(self.cfg.mode.dataset_path)
            
            # Determine if validation split should be enabled
            is_offline_only = self.cfg.env.get("offline_only", False) or self.cfg.env.name in ["mimic", "pyrenees"]
            val_split = self.cfg.get("val_split", None)
            if val_split is None:
                val_split = 0.1 if is_offline_only else 0.0
                
            if val_split > 0 and len(full_reader) > 10:
                self.reader, self.val_reader = full_reader.split(val_ratio=val_split, seed=self.cfg.seed)
            else:
                self.reader = full_reader
                self.val_reader = None
                
            self.train_dataset = OfflineDataset(self.reader)
            if self.val_reader is not None:
                self.val_dataset = OfflineDataset(self.val_reader)
        else:
            # Online mode dummy dataset
            self.train_dataset = torch.utils.data.TensorDataset(torch.zeros(1))
            self.val_dataset = None
            
    def train_dataloader(self):
        batch_size = self.cfg.agent.get("batch_size", 1024) if self.cfg.mode.type == "offline" else self.cfg.agent.get("batch_size", 1)
        default_workers = 2 if torch.cuda.is_available() else 0
        num_workers = self.cfg.get("num_workers", self.cfg.agent.get("num_workers", default_workers))
        pin_memory = self.cfg.get("pin_memory", torch.cuda.is_available())
        persistent_workers = (num_workers > 0)
        if self.reader is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=lambda idxs: self.reader.get_batch(idxs)
            )
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
    
    def val_dataloader(self):
        if self.val_dataset is not None and self.val_reader is not None:
            batch_size = self.cfg.agent.get("batch_size", 1024)
            default_workers = 2 if torch.cuda.is_available() else 0
            num_workers = self.cfg.get("num_workers", self.cfg.agent.get("num_workers", default_workers))
            pin_memory = self.cfg.get("pin_memory", torch.cuda.is_available())
            persistent_workers = (num_workers > 0)
            return DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=lambda idxs: self.val_reader.get_batch(idxs)
            )
        return DataLoader(self.train_dataset, batch_size=1)
