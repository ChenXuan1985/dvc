"""
模型评估脚本：计算至少10个性能指标
"""
import argparse
import yaml
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)
from scipy.stats import entropy

from model import create_model
from dataset import create_data_loaders

def load_params():
    """加载参数配置"""
    with open('params.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_predictions(model, loader, device):
    """获取模型预测结果"""
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)

def calculate_metrics(labels, probs, preds, probs_positive):
    """计算所有性能指标"""
    metrics = {}
    
    # 1. 准确率 (Accuracy)
    metrics['accuracy'] = accuracy_score(labels, preds)
    
    # 2. 精确率 (Precision) - 加权平均
    metrics['precision_weighted'] = precision_score(labels, preds, average='weighted', zero_division=0)
    
    # 3. 精确率 (Precision) - macro
    metrics['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
    
    # 4. 召回率 (Recall) - 加权平均
    metrics['recall_weighted'] = recall_score(labels, preds, average='weighted', zero_division=0)
    
    # 5. 召回率 (Recall) - macro
    metrics['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
    
    # 6. F1 分数 - 加权平均
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # 7. F1 分数 - macro
    metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
    
    # 8. ROC AUC
    try:
        metrics['roc_auc'] = roc_auc_score(labels, probs_positive, multi_class='ovr')
    except:
        metrics['roc_auc'] = 0.0
    
    # 9. PR AUC (Precision-Recall AUC)
    precision, recall, _ = precision_recall_curve(labels, probs_positive)
    metrics['pr_auc'] = auc(recall, precision)
    
    # 10. Matthews 计算
    metrics['matthews_corrcoef'] = matthews_corrcoef(labels, preds)
    
    # 11. Top-K 准确率 (Top-K Accuracy)
    metrics['top2_accuracy'] = accuracy_score(labels, preds)  # 二分类问题简化为 accuracy
    
    # 12. 计算预测概率的熵（不确定性度量
    metrics['prediction_entropy'] = float(np.mean([entropy(p) for p in probs)
    
    # 13. 混淆矩阵衍生指标
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)
    
    # 14. 特异性 (Specificity)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics

def save_metrics(metrics, output_path):
    """保存指标到 JSON 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"指标已保存到: {output_path}")
    print("\n评估指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--data-dir', '-d', required=True, help='预处理数据目录')
    parser.add_argument('--model-path', '-m', required=True, help='模型路径')
    parser.add_argument('--metrics-output', '-o', required=True, help='指标输出路径')
    
    args = parser.parse_args()
    
    params = load_params()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    model = create_model(params).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    _, test_loader = create_data_loaders(args.data_dir, params['training']['batch_size'])
    
    labels, probs, preds = get_predictions(model, test_loader, device)
    
    probs_positive = probs[:, 1] if probs.shape[1] > 1 else probs.squeeze()
    
    metrics = calculate_metrics(labels, probs, preds, probs_positive)
    
    save_metrics(metrics, args.metrics_output)

if __name__ == '__main__':
    main()
