"""
数据生成脚本：生成合成分类数据
"""
import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def load_params():
    """加载参数配置"""
    with open('params.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_synthetic_data(params):
    """生成合成分类数据"""
    data_params = params['data']
    
    X, y = make_classification(
        n_samples=data_params['num_samples'],
        n_features=data_params['num_features'],
        n_classes=data_params['num_classes'],
        n_informative=15,
        n_redundant=5,
        random_state=data_params['random_state'],
        shuffle=True,
        weights=[0.5, 0.5]
    )
    
    # 创建特征名称
    feature_names = [f'feature_{i+1}' for i in range(data_params['num_features'])]
    
    # 转换为 DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame(y, columns=['target'])
    
    return pd.concat([X_df, y_df], axis=1)

def save_data(df, output_path):
    """保存数据到 CSV"""
    df.to_csv(output_path, index=False)
    print(f"数据已保存到: {output_path}")
    print(f"数据形状: {df.shape}")
    print(f"类别分布:\n{df['target'].value_counts()}")

def main():
    parser = argparse.ArgumentParser(description='生成合成分类数据')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    
    args = parser.parse_args()
    
    params = load_params()
    df = generate_synthetic_data(params)
    save_data(df, args.output)

if __name__ == '__main__':
    main()
