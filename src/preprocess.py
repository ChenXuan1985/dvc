import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(input_dir: str) -> tuple:
    input_path = Path(input_dir)
    X = np.load(input_path / "X.npy")
    y = np.load(input_path / "y.npy")
    return X, y


def preprocess_data(X: np.ndarray, y: np.ndarray, params: dict) -> tuple:
    preprocess_params = params["preprocess"]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=preprocess_params["test_size"],
        random_state=preprocess_params["random_state"],
        stratify=y
    )
    
    val_ratio = preprocess_params["val_size"] / (1 - preprocess_params["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=preprocess_params["random_state"],
        stratify=y_train_val
    )
    
    if preprocess_params["normalize"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    output_dir: str
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_data = {
        "X": torch.tensor(X_train, dtype=torch.float32),
        "y": torch.tensor(y_train, dtype=torch.long)
    }
    val_data = {
        "X": torch.tensor(X_val, dtype=torch.float32),
        "y": torch.tensor(y_val, dtype=torch.long)
    }
    test_data = {
        "X": torch.tensor(X_test, dtype=torch.float32),
        "y": torch.tensor(y_test, dtype=torch.long)
    }
    
    torch.save(train_data, output_path / "train.pt")
    torch.save(val_data, output_path / "val.pt")
    torch.save(test_data, output_path / "test.pt")
    
    print(f"Processed data saved to {output_dir}")
    print(f"  - Train: {X_train.shape[0]} samples")
    print(f"  - Val: {X_val.shape[0]} samples")
    print(f"  - Test: {X_test.shape[0]} samples")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    
    params = load_params(args.params)
    X, y = load_raw_data(params["preprocess"]["input_dir"])
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y, params)
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, params["preprocess"]["output_dir"])


if __name__ == "__main__":
    main()
