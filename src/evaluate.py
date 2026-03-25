import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    top_k_accuracy_score,
    confusion_matrix,
    classification_report
)


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    return checkpoint


def load_test_data(data_dir: str):
    data_path = Path(data_dir)
    test_data = torch.load(data_path / "test.pt", weights_only=False)
    return test_data["X"], test_data["y"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, params: dict) -> dict:
    n_classes = y_prob.shape[1]
    
    metrics = {}
    
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    if n_classes == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob[:, 1]))
    else:
        metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        metrics["roc_auc_weighted"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        metrics["pr_auc_macro"] = float(average_precision_score(y_true, y_prob, average="macro"))
        metrics["pr_auc_weighted"] = float(average_precision_score(y_true, y_prob, average="weighted"))
    
    metrics["log_loss"] = float(log_loss(y_true, y_prob))
    
    for k in params["evaluate"]["top_k"]:
        if k <= n_classes:
            metrics[f"top_{k}_accuracy"] = float(top_k_accuracy_score(y_true, y_prob, k=k))
    
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    metrics["confusion_matrix_normalized"] = cm_normalized.tolist()
    
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    
    specificity = np.mean(tn / (tn + fp + 1e-10))
    metrics["specificity_macro"] = float(specificity)
    
    balanced_acc = (metrics["recall_macro"] + specificity) / 2
    metrics["balanced_accuracy"] = float(balanced_acc)
    
    return metrics


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    n_classes = y_prob.shape[1]
    
    per_class = {}
    
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        y_prob_binary = y_prob[:, i]
        
        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
        
        per_class[f"class_{i}"] = {
            "precision": float(tp / (tp + fp + 1e-10)),
            "recall": float(tp / (tp + fn + 1e-10)),
            "f1": float(2 * tp / (2 * tp + fp + fn + 1e-10)),
            "specificity": float(tn / (tn + fp + 1e-10)),
            "support": int(np.sum(y_true_binary))
        }
    
    return per_class


def evaluate(params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_test, y_test = load_test_data(params["evaluate"]["data_dir"])
    X_test = X_test.to(device)
    y_test_np = y_test.numpy()
    
    checkpoint = load_model(params["evaluate"]["model_path"], device)
    
    hidden_sizes = params["train"]["model"]["hidden_sizes"]
    dropout_rate = params["train"]["model"]["dropout_rate"]
    activation = params["train"]["model"]["activation"]
    input_size = X_test.shape[1]
    num_classes = len(torch.unique(y_test))
    
    from train import ClassificationModel
    model = ClassificationModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        activation=activation
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test)
        y_prob = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred = np.argmax(y_prob, axis=1)
    
    metrics = compute_metrics(y_test_np, y_pred, y_prob, params)
    per_class_metrics = compute_per_class_metrics(y_test_np, y_pred, y_prob)
    
    results = {
        "overall_metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "model_info": {
            "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
            "checkpoint_metrics": checkpoint.get("metrics", {})
        },
        "dataset_info": {
            "test_samples": len(y_test_np),
            "num_classes": num_classes,
            "num_features": input_size
        }
    }
    
    metrics_dir = Path(params["evaluate"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    dvc_metrics = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision_macro"],
        "recall": metrics["recall_macro"],
        "f1": metrics["f1_macro"],
        "roc_auc": metrics.get("roc_auc", metrics.get("roc_auc_macro", 0)),
        "pr_auc": metrics.get("pr_auc", metrics.get("pr_auc_macro", 0)),
        "log_loss": metrics["log_loss"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "specificity": metrics["specificity_macro"],
        "top_3_accuracy": metrics.get("top_3_accuracy", 0)
    }
    
    with open(metrics_dir / "dvc_metrics.json", "w") as f:
        json.dump(dvc_metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1 Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  Specificity:        {metrics['specificity_macro']:.4f}")
    print(f"  Log Loss:           {metrics['log_loss']:.4f}")
    
    if "roc_auc" in metrics:
        print(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
        print(f"  PR AUC:             {metrics['pr_auc']:.4f}")
    else:
        print(f"  ROC AUC (Macro):    {metrics['roc_auc_macro']:.4f}")
        print(f"  PR AUC (Macro):     {metrics['pr_auc_macro']:.4f}")
    
    if "top_3_accuracy" in metrics:
        print(f"  Top-3 Accuracy:     {metrics['top_3_accuracy']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in per_class_metrics.items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}, Recall: {class_metrics['recall']:.4f}, "
              f"F1: {class_metrics['f1']:.4f}, Support: {class_metrics['support']}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(cm)
    
    print(f"\nResults saved to {metrics_dir / 'metrics.json'}")
    print(f"DVC metrics saved to {metrics_dir / 'dvc_metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification model")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    
    params = load_params(args.params)
    evaluate(params)


if __name__ == "__main__":
    main()
