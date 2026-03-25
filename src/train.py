"""
模型训练脚本
使用 PyTorch 构建和训练分类模型
"""
import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
import yaml
from tqdm import tqdm


class ClassificationDataset(Dataset):
    """分类数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """多层感知机分类器"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3, activation="relu"):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.0001, monitor="val_loss", mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.is_better(value, self.best_value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def load_params(params_path="params.yaml"):
    """加载参数配置"""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_dir="data"):
    """加载预处理后的数据"""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    return X_train, X_val, y_train, y_val


def create_dataloaders(X_train, X_val, y_train, y_val, batch_size=64):
    """创建数据加载器"""
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_model(model_params, device):
    """创建模型"""
    model = MLPClassifier(
        input_dim=model_params["input_dim"],
        hidden_dims=model_params["hidden_dims"],
        output_dim=model_params["output_dim"],
        dropout_rate=model_params["dropout_rate"],
        activation=model_params["activation"]
    )
    return model.to(device)


def create_optimizer(model, train_params):
    """创建优化器"""
    optimizer_name = train_params["optimizer"].lower()
    lr = train_params["learning_rate"]
    weight_decay = train_params["weight_decay"]
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"未知的优化器: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, train_params):
    """创建学习率调度器"""
    scheduler_name = train_params.get("scheduler", "reduce_on_plateau")
    
    if scheduler_name == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=train_params.get("scheduler_patience", 5),
            factor=train_params.get("scheduler_factor", 0.5),
            verbose=True
        )
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_name == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, is_best=False):
    """保存模型检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    if is_best:
        best_path = checkpoint_path.replace(".pt", "_best.pt")
        torch.save(checkpoint, best_path)


def save_training_history(history, output_dir="metrics"):
    """保存训练历史"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--params", type=str, default="params.yaml", help="参数配置文件路径")
    parser.add_argument("--data-dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output-dir", type=str, default="models", help="模型输出目录")
    parser.add_argument("--metrics-dir", type=str, default="metrics", help="指标输出目录")
    args = parser.parse_args()
    
    # 加载参数
    params = load_params(args.params)
    model_params = params["model"]
    train_params = params["train"]
    early_stopping_params = params["early_stopping"]
    checkpoint_params = params["checkpoint"]
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    X_train, X_val, y_train, y_val = load_data(args.data_dir)
    train_loader, val_loader = create_dataloaders(
        X_train, X_val, y_train, y_val,
        batch_size=train_params["batch_size"]
    )
    
    # 创建模型
    model = create_model(model_params, device)
    print(f"模型结构:\n{model}")
    
    # 创建优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, train_params)
    scheduler = create_scheduler(optimizer, train_params)
    
    # 早停
    early_stopping = None
    if early_stopping_params["enabled"]:
        early_stopping = EarlyStopping(
            patience=early_stopping_params["patience"],
            min_delta=early_stopping_params["min_delta"],
            monitor=early_stopping_params["monitor"],
            mode=early_stopping_params["mode"]
        )
    
    # 训练循环
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    best_val_metric = float("inf") if checkpoint_params["mode"] == "min" else float("-inf")
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    
    print(f"\n开始训练 {train_params['epochs']} 个 epochs...")
    for epoch in range(train_params["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{train_params['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 保存最佳模型
        monitor_value = val_acc if checkpoint_params["monitor"] == "val_accuracy" else val_loss
        is_best = False
        
        if checkpoint_params["mode"] == "max":
            if monitor_value > best_val_metric:
                best_val_metric = monitor_value
                is_best = True
        else:
            if monitor_value < best_val_metric:
                best_val_metric = monitor_value
                is_best = True
        
        if is_best or not checkpoint_params["save_best_only"]:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
            save_checkpoint(model, optimizer, epoch + 1, metrics, checkpoint_path, is_best)
        
        # 早停检查
        if early_stopping is not None:
            stop_value = val_loss if early_stopping_params["monitor"] == "val_loss" else val_acc
            if early_stopping(stop_value):
                print(f"\n早停触发! 在 epoch {epoch+1} 停止训练")
                break
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n最终模型已保存到 {final_model_path}")
    
    # 保存训练历史
    save_training_history(history, args.metrics_dir)
    
    # 保存训练摘要
    summary = {
        "total_epochs": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "best_val_metric": best_val_metric,
        "early_stopped": early_stopping.early_stop if early_stopping else False
    }
    
    with open(os.path.join(args.metrics_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n训练完成!")
    print(f"最佳验证指标: {best_val_metric:.4f}")


if __name__ == "__main__":
    main()
