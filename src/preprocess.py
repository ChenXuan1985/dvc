"""
数据预处理脚本：数据清洗、标准化和划分
"""
import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_params():
    """加载参数配置"""
    with open('params.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data(input_path):
    """加载原始数据"""
    return pd.read_csv(input_path)

def split_features_target(df):
    """分离特征和目标变量"""
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def normalize_features(X_train, X_test, params):
    """特征标准化"""
    if params['data']['normalize']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train.values, X_test.values, None

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir):
    """保存预处理后的数据"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_test.npy', y_test)
    
    print(f"预处理数据已保存到: {output_dir}")
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

def main():
    parser = argparse.ArgumentParser(description='数据预处理')
    parser.add_argument('--input', '-i', required=True, help='输入数据路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录路径')
    
    args = parser.parse_args()
    
    params = load_params()
    
    df = load_data(args.input)
    
    X, y = split_features_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state'],
        stratify=y
    )
    
    X_train, X_test, scaler = normalize_features(X_train, X_test, params)
    
    save_preprocessed_data(X_train, X_test, y_train.values, y_test.values, args.output)

if __name__ == '__main__':
    main()
