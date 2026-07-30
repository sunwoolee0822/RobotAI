import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class RobotAIDataset(Dataset):
    def __init__(self, data_root, indices):
        self.data   = torch.from_numpy(np.load(data_root + 'array.npy')[indices]).float()
        self.time   = torch.from_numpy(np.load(data_root + 'time.npy')[indices]).float()
        self.static = torch.from_numpy(np.load(data_root + 'static.npy')[indices]).float()
        self.mask   = torch.from_numpy(np.load(data_root + 'mask.npy')[indices]).int()
        self.gt     = torch.from_numpy(np.load(data_root + 'gt.npy')[indices]).long()

        print(f"  data:   {self.data.shape}")
        print(f"  time:   {self.time.shape}")
        print(f"  static: {self.static.shape}")
        print(f"  mask:   {self.mask.shape}")
        print(f"  gt:     {self.gt.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data':   self.data[idx],
            'time':   self.time[idx],
            'static': self.static[idx],
            'mask':   self.mask[idx],
            'gt':     self.gt[idx],
        }
    
    def get_sampler(self):
        """Train용 WeightedRandomSampler (1:1 oversampling)"""
        labels = self.gt.squeeze().numpy()
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(weights),
            replacement=True,
        )
        return sampler

class RobotAIDataModule(pl.LightningDataModule):
    def __init__(self, data_root, split_path, batch_size=32, num_workers=4):
        super().__init__()
        self.data_root   = data_root
        self.split_path  = split_path
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        split = np.load(self.split_path, allow_pickle=True)
        print(f"[Train] {len(split[0])} samples")
        self.train_dset = RobotAIDataset(self.data_root, split[0])
        print(f"[Val]   {len(split[1])} samples")
        self.val_dset   = RobotAIDataset(self.data_root, split[1])
        print(f"[Test]  {len(split[2])} samples")
        self.test_dset  = RobotAIDataset(self.data_root, split[2])

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size,
                          sampler=self.train_dset.get_sampler(), num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dset,   batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dset,  batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)
