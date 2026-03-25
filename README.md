# DVC + PyTorch ML 流水线工程化项目

基于 DVC 和 PyTorch 的完整机器学习流水线，实现数据版本控制、流水线管理和实验追踪。

## 项目结构

```
├── data/
│   ├── raw/          # 原始数据
│   └── processed/   # 预处理后的数据
├── models/
│   ├── checkpoints/  # 模型检查点
│   └── model.pt    # 训练好的模型
├── reports/         # 评估报告和指标
├── logs/            # 训练日志
├── src/             # 源代码
│   ├── generate_data.py   # 数据生成
│   ├── preprocess.py    # 数据预处理
│   ├── model.py         # PyTorch 模型定义
│   ├── dataset.py       # 数据集加载器
│   ├── train.py         # 模型训练
│   └── evaluate.py      # 模型评估
├── params.yaml      # 超参数配置
├── dvc.yaml         # DVC 流水线定义
├── requirements.txt  # Python 依赖
├── .gitignore     # Git 忽略规则
└── .dvcignore       # DVC 忽略规则
└── README.md         # 使用说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化 DVC（如果需要）

```bash
dvc init
```

### 3. 运行完整流水线

```bash
dvc repro
```

### 4. 查看流水线状态

```bash
dvc status
```

### 5. 查看指标

```bash
dvc metrics show
```

## 流水线阶段

### 1. 数据生成 (generate_data)

生成合成分类数据集。

```bash
dvc repro generate_data
```

### 2. 数据预处理 (preprocess)

数据清洗、特征标准化、训练/测试集划分。

```bash
dvc repro preprocess
```

### 3. 模型训练 (train)

使用 PyTorch 训练分类模型，支持早停策略。

```bash
dvc repro train
```

### 4. 模型评估 (evaluate)

计算并保存14个性能指标。

```bash
dvc repro evaluate
```

## 实验追踪

### 创建实验

```bash
dvc exp run
```

### 查看实验列表

```bash
dvc exp show
```

### 应用实验结果

```bash
dvc exp apply <exp-id>
```

### 推送实验到远程

```bash
dvc exp push
```

## 配置参数

所有超参数配置在 `params.yaml`：

- `data`: 数据参数（样本数、特征数、类别数、测试集比例
- `model`: 模型结构（隐藏层、Dropout
- `training`: 训练参数（批次大小、epoch 数、学习率、早停配置

## 性能指标

评估阶段计算以下指标：

1. **Accuracy** - 准确率
2. **Precision (weighted)** - 加权精确率
3. **Precision (macro)** - 宏平均精确率
4. **Recall (weighted)** - 加权召回率
5. **Recall (macro)** - 宏平均召回率
6. **F1 (weighted)** - 加权 F1 分数
7. **F1 (macro)** - 宏平均 F1 分数
8. **ROC AUC** - ROC 曲线下面积
9. **PR AUC** - Precision-Recall 曲线下面积
10. **Matthews Correlation Coefficient** - Matthews 相关系数
11. **Top-2 Accuracy** - Top-2 准确率
12. **Prediction Entropy** - 预测熵（不确定性度量）
13. **Specificity** - 特异性
14. 混淆矩阵指标（TN, FP, FN, TP)**

## 远程存储配置

### 添加远程存储（以 S3 为例）

```bash
dvc remote add -d myremote s3://bucket/path
```

### 推送数据到远程

```bash
dvc push
```

### 从远程拉取数据

```bash
dvc pull
```

## 许可证

Apache License 2.0
