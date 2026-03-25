"""
PyTorch 数据集和数据加载器
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """自定义数据集"""
    def __init__(self, X_path, y_path):
        self.X = torch.tensor(np.load(X_path, allow_pickle=True).astype(np.float32)
        self.y = torch.tensor(np.load(y_path, allow_pickle=True)).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(data_dir, batch_size, num_workers=0):
    """创建数据加载器"""
    train_dataset = CustomDataset(
        f'{data_dir}/X_train.npy', f'{data_dir}/y_train.npy')
    
    test_dataset = CustomDataset(
        f'{data_dir}/X_test.npy', f'{data_dir}/y_test.npy')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def create_train_loader(data_dir, batch_size, num_workers=0):
    """仅创建训练集加载器（用于验证阶段）"""
    train_dataset = CustomDataset(
        f'{data_dir}/X_train.npy', f'{data_dir}/y_train.npy')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return train_loader
