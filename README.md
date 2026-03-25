# ML Pipeline with DVC and PyTorch

基于 DVC 和 PyTorch 的机器学习流水线工程化项目。

## 项目结构

```
ml-dvc-pipeline/
├── data/
│   ├── raw/                    # 原始数据 (DVC 跟踪)
│   │   ├── X.npy
│   │   └── y.npy
│   └── processed/              # 预处理后数据 (DVC 跟踪)
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
├── models/                     # 模型文件 (DVC 跟踪)
│   ├── best_model.pt
│   └── training_history.json
├── metrics/                    # 评估指标
│   ├── metrics.json
│   └── dvc_metrics.json
├── src/                        # 源代码
│   ├── generate_data.py        # 数据生成
│   ├── preprocess.py           # 数据预处理
│   ├── train.py                # 模型训练
│   └── evaluate.py             # 模型评估
├── dvc.yaml                    # DVC 流水线定义
├── params.yaml                 # 超参数配置
├── requirements.txt            # Python 依赖
├── .gitignore
└── .dvcignore
```

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 初始化 DVC (如果尚未初始化)
dvc init
```

### 2. 运行完整流水线

```bash
# 一键运行整个流水线
dvc repro
```

### 3. 查看实验结果

```bash
# 查看指标
dvc metrics show

# 比较实验
dvc exp show
```

## 流水线阶段

### Stage 1: 数据生成 (`generate_data`)

生成合成分类数据集。

**输入**: params.yaml 中的数据配置参数

**输出**: 
- `data/raw/X.npy` - 特征矩阵
- `data/raw/y.npy` - 标签向量

### Stage 2: 数据预处理 (`preprocess`)

数据标准化、训练/验证/测试集划分。

**输入**: 
- 原始数据 (X.npy, y.npy)
- 预处理参数

**输出**:
- `data/processed/train.pt`
- `data/processed/val.pt`
- `data/processed/test.pt`

### Stage 3: 模型训练 (`train`)

PyTorch 分类模型训练，包含早停和模型检查点。

**特性**:
- 可配置的网络架构 (隐藏层大小、Dropout、激活函数)
- 支持多种优化器 (Adam, AdamW, SGD)
- 学习率调度器
- 早停机制
- 最佳模型保存

**输出**:
- `models/best_model.pt` - 最佳模型检查点
- `models/training_history.json` - 训练历史

### Stage 4: 模型评估 (`evaluate`)

计算 10+ 评估指标。

**指标列表**:
1. **Accuracy** - 准确率
2. **Precision (Macro/Weighted)** - 精确率
3. **Recall (Macro/Weighted)** - 召回率
4. **F1 Score (Macro/Weighted)** - F1 分数
5. **ROC AUC** - ROC 曲线下面积
6. **PR AUC** - PR 曲线下面积
7. **Log Loss** - 对数损失
8. **Balanced Accuracy** - 平衡准确率
9. **Specificity** - 特异性
10. **Top-K Accuracy** - Top-K 准确率

**输出**:
- `metrics/metrics.json` - 完整评估报告
- `metrics/dvc_metrics.json` - DVC 跟踪的指标

## 参数配置

所有超参数在 `params.yaml` 中配置：

```yaml
data:
  n_samples: 2000        # 样本数量
  n_features: 20         # 特征数量
  n_classes: 3           # 类别数量

train:
  model:
    hidden_sizes: [128, 64, 32]  # 隐藏层大小
    dropout_rate: 0.3            # Dropout 比率
  training:
    epochs: 100                  # 训练轮数
    batch_size: 64               # 批次大小
    early_stopping:
      patience: 10               # 早停耐心值
```

## 实验管理

### 运行新实验

```bash
# 修改 params.yaml 中的参数后运行
dvc exp run

# 或使用参数覆盖
dvc exp run --set-param train.optimizer.lr=0.01
```

### 查看和比较实验

```bash
# 列出所有实验
dvc exp list

# 显示实验对比
dvc exp show

# 应用最佳实验
dvc exp apply <exp-name>
```

### 持久化实验

```bash
# 保存实验到 Git
dvc exp save
```

## 数据管理

### 使用 DVC 跟踪数据

```bash
# 跟踪数据文件
dvc add data/raw/X.npy data/raw/y.npy

# 推送到远程存储
dvc push
```

### 配置远程存储

```bash
# 添加远程存储
dvc remote add -d myremote s3://mybucket/dvc-storage

# 推送数据
dvc push
```

## 模型架构

```
ClassificationModel(
  (network): Sequential(
    Linear(input_size, 128)
    BatchNorm1d(128)
    ReLU()
    Dropout(0.3)
    Linear(128, 64)
    BatchNorm1d(64)
    ReLU()
    Dropout(0.3)
    Linear(64, 32)
    BatchNorm1d(32)
    ReLU()
    Dropout(0.3)
    Linear(32, num_classes)
  )
)
```

## 开发指南

### 单独运行某个阶段

```bash
# 只运行数据生成
dvc repro generate_data

# 从预处理开始运行
dvc repro preprocess
```

### 强制重新运行

```bash
# 强制重新运行所有阶段
dvc repro --force
```

### 检查流水线状态

```bash
# 查看流水线状态
dvc status

# 可视化 DAG
dvc dag
```

## 许可证

Apache 2.0
