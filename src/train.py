import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ClassificationModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, num_classes: int, dropout_rate: float, activation: str = "relu"):
        super(ClassificationModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        activation_fn = nn.ReLU() if activation == "relu" else nn.GELU()
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                activation_fn,
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, monitor: str = "val_loss", mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == "min":
            self.compare = lambda current, best: current < best - min_delta
        else:
            self.compare = lambda current, best: current > best + min_delta
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ModelCheckpoint:
    def __init__(self, save_dir: str, monitor: str = "val_loss", mode: str = "min", save_best_only: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
        if mode == "min":
            self.compare = lambda current, best: current < best
        else:
            self.compare = lambda current, best: current > best
    
    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, score: float, metrics: dict):
        if self.save_best_only:
            if self.best_score is None or self.compare(score, self.best_score):
                self.best_score = score
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics
                }
                torch.save(checkpoint, self.save_dir / "best_model.pt")
                return True
            return False
        else:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics
            }
            torch.save(checkpoint, self.save_dir / f"model_epoch_{epoch}.pt")
            return True


def load_data(data_dir: str) -> tuple:
    data_path = Path(data_dir)
    
    train_data = torch.load(data_path / "train.pt")
    val_data = torch.load(data_path / "val.pt")
    
    train_dataset = TensorDataset(train_data["X"], train_data["y"])
    val_dataset = TensorDataset(val_data["X"], val_data["y"])
    
    return train_dataset, val_dataset, train_data["X"].shape[1], len(torch.unique(train_data["y"]))


def create_optimizer(model: nn.Module, params: dict) -> torch.optim.Optimizer:
    opt_params = params["train"]["optimizer"]
    
    if opt_params["name"] == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_params["lr"],
            weight_decay=opt_params["weight_decay"]
        )
    elif opt_params["name"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_params["lr"],
            weight_decay=opt_params["weight_decay"]
        )
    elif opt_params["name"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_params["lr"],
            weight_decay=opt_params["weight_decay"],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_params['name']}")


def create_scheduler(optimizer: torch.optim.Optimizer, params: dict):
    scheduler_params = params["train"]["scheduler"]
    
    if scheduler_params["name"] == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_params["factor"],
            patience=scheduler_params["patience"]
        )
    elif scheduler_params["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params["train"]["training"]["epochs"]
        )
    else:
        return None


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(params: dict):
    set_seed(params["train"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset, val_dataset, input_size, num_classes = load_data(params["train"]["input_dir"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["train"]["training"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["train"]["training"]["batch_size"],
        shuffle=False
    )
    
    model = ClassificationModel(
        input_size=input_size,
        hidden_sizes=params["train"]["model"]["hidden_sizes"],
        num_classes=num_classes,
        dropout_rate=params["train"]["model"]["dropout_rate"],
        activation=params["train"]["model"]["activation"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)
    
    early_stopping = EarlyStopping(
        patience=params["train"]["training"]["early_stopping"]["patience"],
        min_delta=params["train"]["training"]["early_stopping"]["min_delta"],
        monitor=params["train"]["training"]["early_stopping"]["monitor"]
    )
    
    checkpoint = ModelCheckpoint(
        save_dir=params["train"]["model_dir"],
        monitor=params["train"]["training"]["checkpoint"]["monitor"],
        save_best_only=params["train"]["training"]["checkpoint"]["save_best_only"]
    )
    
    epochs = params["train"]["training"]["epochs"]
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch + 1
        }
        
        is_best = checkpoint.save(model, optimizer, epoch + 1, val_loss, metrics)
        
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
              f"{' [BEST]' if is_best else ''}")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    history_path = Path(params["train"]["model_dir"]) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best model saved to {params['train']['model_dir']}/best_model.pt")
    print(f"Training history saved to {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    
    params = load_params(args.params)
    train(params)


if __name__ == "__main__":
    main()
