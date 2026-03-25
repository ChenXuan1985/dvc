"""
数据生成脚本
生成用于分类任务的合成数据集
"""
import argparse
import json
import os

import numpy as np
import yaml
from sklearn.datasets import make_classification


def load_params(params_path="params.yaml"):
    """加载参数配置"""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def generate_classification_data(params):
    """生成分类数据集"""
    data_params = params["data"]
    
    X, y = make_classification(
        n_samples=data_params["n_samples"],
        n_features=data_params["n_features"],
        n_informative=data_params["n_informative"],
        n_redundant=data_params["n_redundant"],
        n_classes=data_params["n_classes"],
        random_state=data_params["random_state"],
        n_clusters_per_class=2,
        class_sep=1.5,
    )
    
    return X, y


def save_raw_data(X, y, output_dir="data"):
    """保存原始数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X_raw.npy"), X)
    np.save(os.path.join(output_dir, "y_raw.npy"), y)
    
    # 保存数据信息
    data_info = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    }
    
    with open(os.path.join(output_dir, "data_info.json"), "w") as f:
        json.dump(data_info, f, indent=2)
    
    print(f"原始数据已保存到 {output_dir}")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")


def main():
    parser = argparse.ArgumentParser(description="生成分类数据集")
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="参数配置文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="输出目录"
    )
    args = parser.parse_args()
    
    # 加载参数
    params = load_params(args.params)
    
    # 生成数据
    X, y = generate_classification_data(params)
    
    # 保存数据
    save_raw_data(X, y, args.output_dir)


if __name__ == "__main__":
    main()
