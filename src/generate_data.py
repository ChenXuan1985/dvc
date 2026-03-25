import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.datasets import make_classification


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def generate_data(params: dict) -> tuple:
    data_params = params["data"]
    
    X, y = make_classification(
        n_samples=data_params["n_samples"],
        n_features=data_params["n_features"],
        n_classes=data_params["n_classes"],
        n_informative=data_params["n_informative"],
        n_redundant=data_params["n_redundant"],
        random_state=data_params["random_state"],
        shuffle=True,
    )
    
    return X, y


def save_data(X: np.ndarray, y: np.ndarray, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "X.npy", X)
    np.save(output_path / "y.npy", y)
    
    print(f"Data saved to {output_dir}")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - Classes: {np.unique(y)}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    
    params = load_params(args.params)
    X, y = generate_data(params)
    save_data(X, y, params["data"]["output_dir"])


if __name__ == "__main__":
    main()
