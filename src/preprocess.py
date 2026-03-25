"""
数据预处理脚本
包括数据标准化、训练/验证/测试集划分
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_params(params_path="params.yaml"):
    """加载参数配置"""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(data_dir="data"):
    """加载原始数据"""
    X = np.load(os.path.join(data_dir, "X_raw.npy"))
    y = np.load(os.path.join(data_dir, "y_raw.npy"))
    return X, y


def normalize_data(X_train, X_val, X_test, method="standard"):
    """数据标准化"""
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"未知的标准化方法: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                        scaler, output_dir="data"):
    """保存处理后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 NumPy 数组
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    # 保存 scaler 参数
    scaler_params = {
        "method": "standard" if isinstance(scaler, StandardScaler) else "minmax",
        "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
        "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
        "data_min": scaler.data_min_.tolist() if hasattr(scaler, "data_min_") else None,
        "data_max": scaler.data_max_.tolist() if hasattr(scaler, "data_max_") else None,
    }
    
    with open(os.path.join(output_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)
    
    # 保存预处理信息
    preprocess_info = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "n_classes": len(np.unique(y_train)),
        "train_class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "val_class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
        "test_class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
    }
    
    with open(os.path.join(output_dir, "preprocess_info.json"), "w") as f:
        json.dump(preprocess_info, f, indent=2)
    
    print(f"预处理数据已保存到 {output_dir}")
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")


def main():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="参数配置文件路径"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="输入数据目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="输出数据目录"
    )
    args = parser.parse_args()
    
    # 加载参数
    params = load_params(args.params)
    data_params = params["data"]
    preprocess_params = params["preprocess"]
    
    # 加载原始数据
    X, y = load_raw_data(args.input_dir)
    
    # 首先划分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=data_params["test_size"],
        random_state=data_params["random_state"],
        stratify=y
    )
    
    # 从剩余数据中划分训练集和验证集
    val_ratio = data_params["val_size"] / (1 - data_params["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=data_params["random_state"],
        stratify=y_temp
    )
    
    # 数据标准化
    if preprocess_params["normalize"]:
        X_train, X_val, X_test, scaler = normalize_data(
            X_train, X_val, X_test,
            method=preprocess_params["normalization_method"]
        )
    else:
        scaler = None
    
    # 保存处理后的数据
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                        scaler, args.output_dir)


if __name__ == "__main__":
    main()
