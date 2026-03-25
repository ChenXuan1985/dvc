"""
模型评估脚本
计算分类模型的多个性能指标
"""
import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, top_k_accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, log_loss, jaccard_score
)


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


def load_params(params_path="params.yaml"):
    """加载参数配置"""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_test_data(data_dir="data"):
    """加载测试数据"""
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_test, y_test


def load_model(model_path, model_params, device):
    """加载训练好的模型"""
    model = MLPClassifier(
        input_dim=model_params["input_dim"],
        hidden_dims=model_params["hidden_dims"],
        output_dim=model_params["output_dim"],
        dropout_rate=model_params["dropout_rate"],
        activation=model_params["activation"]
    )
    
    # 加载模型权重
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    return model.to(device)


def get_predictions(model, X_test, batch_size=128, device="cpu"):
    """获取模型预测结果"""
    model.eval()
    dataset = ClassificationDataset(X_test, np.zeros(len(X_test)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    return y_pred, y_prob


def calculate_metrics(y_true, y_pred, y_prob, n_classes, top_k=3) -> Dict[str, Any]:
    """计算所有评估指标（至少10个）"""
    metrics = {}
    
    # 1. 准确率 (Accuracy)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    # 2. 平衡准确率 (Balanced Accuracy)
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    
    # 3. 精确率 (Precision) - 宏平均
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    
    # 4. 召回率 (Recall) - 宏平均
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    
    # 5. F1 分数 - 宏平均
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    # 6. ROC AUC (多分类使用 OvR 策略)
    try:
        if n_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            metrics["roc_auc_ovo"] = float(roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro"))
    except ValueError:
        metrics["roc_auc"] = None
    
    # 7. PR AUC (Average Precision)
    try:
        if n_classes == 2:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            metrics["pr_auc_macro"] = float(average_precision_score(y_true, y_prob, average="macro"))
    except ValueError:
        metrics["pr_auc"] = None
    
    # 8. Top-K 准确率
    if n_classes >= top_k:
        try:
            metrics[f"top_{top_k}_accuracy"] = float(top_k_accuracy_score(y_true, y_prob, k=top_k))
        except ValueError:
            metrics[f"top_{top_k}_accuracy"] = None
    
    # 9. Cohen's Kappa
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    
    # 10. Matthews Correlation Coefficient
    metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    
    # 11. Jaccard Score
    metrics["jaccard_macro"] = float(jaccard_score(y_true, y_pred, average="macro", zero_division=0))
    
    # 12. 对数损失 (Log Loss)
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
    except ValueError:
        metrics["log_loss"] = None
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # 每个类别的指标
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["per_class_metrics"] = class_report
    
    return metrics


def save_metrics(metrics: Dict[str, Any], output_dir="metrics"):
    """保存评估指标"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有指标
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 保存简化版指标（用于 DVC 追踪）
    summary_metrics = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "cohen_kappa": metrics["cohen_kappa"],
        "matthews_corrcoef": metrics["matthews_corrcoef"],
    }
    
    # 添加 ROC AUC
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        summary_metrics["roc_auc"] = metrics["roc_auc"]
    elif "roc_auc_ovr" in metrics and metrics["roc_auc_ovr"] is not None:
        summary_metrics["roc_auc"] = metrics["roc_auc_ovr"]
    
    # 添加 PR AUC
    if "pr_auc" in metrics and metrics["pr_auc"] is not None:
        summary_metrics["pr_auc"] = metrics["pr_auc"]
    elif "pr_auc_macro" in metrics and metrics["pr_auc_macro"] is not None:
        summary_metrics["pr_auc"] = metrics["pr_auc_macro"]
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)
    
    return summary_metrics


def generate_report(metrics: Dict[str, Any], output_dir="reports"):
    """生成评估报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_lines = [
        "# 模型评估报告\n",
        "## 主要指标\n",
        f"- **准确率 (Accuracy)**: {metrics['accuracy']:.4f}",
        f"- **平衡准确率 (Balanced Accuracy)**: {metrics['balanced_accuracy']:.4f}",
        f"- **精确率 (Precision Macro)**: {metrics['precision_macro']:.4f}",
        f"- **召回率 (Recall Macro)**: {metrics['recall_macro']:.4f}",
        f"- **F1 分数 (Macro)**: {metrics['f1_macro']:.4f}",
        "",
        "## 高级指标",
    ]
    
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        report_lines.append(f"- **ROC AUC**: {metrics['roc_auc']:.4f}")
    if "roc_auc_ovr" in metrics and metrics["roc_auc_ovr"] is not None:
        report_lines.append(f"- **ROC AUC (OvR)**: {metrics['roc_auc_ovr']:.4f}")
    
    if "pr_auc" in metrics and metrics["pr_auc"] is not None:
        report_lines.append(f"- **PR AUC**: {metrics['pr_auc']:.4f}")
    if "pr_auc_macro" in metrics and metrics["pr_auc_macro"] is not None:
        report_lines.append(f"- **PR AUC (Macro)**: {metrics['pr_auc_macro']:.4f}")
    
    report_lines.extend([
        f"- **Cohen's Kappa**: {metrics['cohen_kappa']:.4f}",
        f"- **Matthews Correlation Coefficient**: {metrics['matthews_corrcoef']:.4f}",
        f"- **Jaccard Score (Macro)**: {metrics['jaccard_macro']:.4f}",
        "",
        "## 混淆矩阵",
        "```",
    ])
    
    # 添加混淆矩阵
    cm = np.array(metrics["confusion_matrix"])
    for row in cm:
        report_lines.append("  ".join([f"{x:4d}" for x in row]))
    
    report_lines.append("```")
    
    # 保存报告
    with open(os.path.join(output_dir, "evaluation_report.md"), "w") as f:
        f.write("\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--params", type=str, default="params.yaml", help="参数配置文件路径")
    parser.add_argument("--data-dir", type=str, default="data", help="数据目录")
    parser.add_argument("--model-dir", type=str, default="models", help="模型目录")
    parser.add_argument("--metrics-dir", type=str, default="metrics", help="指标输出目录")
    parser.add_argument("--reports-dir", type=str, default="reports", help="报告输出目录")
    args = parser.parse_args()
    
    # 加载参数
    params = load_params(args.params)
    model_params = params["model"]
    eval_params = params["eval"]
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载测试数据
    X_test, y_test = load_test_data(args.data_dir)
    print(f"测试集大小: {len(X_test)}")
    
    # 加载模型
    model_path = os.path.join(args.model_dir, "checkpoint_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, "checkpoint.pt")
    
    model = load_model(model_path, model_params, device)
    print(f"已加载模型: {model_path}")
    
    # 获取预测
    y_pred, y_prob = get_predictions(model, X_test, eval_params["batch_size"], device)
    
    # 计算指标
    print("\n计算评估指标...")
    metrics = calculate_metrics(
        y_test, y_pred, y_prob,
        n_classes=model_params["output_dim"],
        top_k=eval_params.get("top_k", 3)
    )
    
    # 保存指标
    summary_metrics = save_metrics(metrics, args.metrics_dir)
    
    # 生成报告
    generate_report(metrics, args.reports_dir)
    
    # 打印主要指标
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    for key, value in summary_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key:25s}: {value:.4f}")
    print("="*50)
    
    print(f"\n详细指标已保存到 {args.metrics_dir}/evaluation_metrics.json")
    print(f"简化指标已保存到 {args.metrics_dir}/metrics.json")
    print(f"评估报告已保存到 {args.reports_dir}/evaluation_report.md")


if __name__ == "__main__":
    main()
