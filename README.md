# 基于 DVC 和 PyTorch 的 ML 流水线

这是一个完整的、可复现的机器学习流水线项目，使用 PyTorch 构建分类模型，并通过 DVC 实现数据版本控制、流水线管理和实验追踪。

## 项目结构

```
.
├── data/                   # 数据目录（由 DVC 管理）
│   ├── X_raw.npy          # 原始特征
│   ├── y_raw.npy          # 原始标签
│   ├── X_train.npy        # 训练集
│   ├── X_val.npy          # 验证集
│   ├── X_test.npy         # 测试集
│   └── ...
├── src/                    # 源代码
│   ├── generate_data.py   # 数据生成
│   ├── preprocess.py      # 数据预处理
│   ├── train.py           # 模型训练
│   └── evaluate.py        # 模型评估
├── models/                 # 模型目录（由 DVC 管理）
│   ├── model.pt           # 最终模型
│   ├── checkpoint.pt      # 检查点
│   └── checkpoint_best.pt # 最佳模型
├── metrics/                # 指标目录
│   ├── metrics.json       # 主要指标（DVC 追踪）
│   ├── evaluation_metrics.json  # 详细指标
│   ├── training_history.json    # 训练历史
│   └── train_summary.json       # 训练摘要
├── reports/                # 报告目录
│   └── evaluation_report.md     # 评估报告
├── dvc.yaml               # DVC 流水线配置
├── params.yaml            # 超参数配置
├── requirements.txt       # Python 依赖
├── .gitignore            # Git 忽略规则
└── .dvcignore            # DVC 忽略规则
```

## 核心特性

### L1-OPS (工程化标准)

✅ **流水线稳定可复现**
- 完整的 DVC 流水线定义 (`dvc.yaml`)
- 所有依赖和参数可追踪
- 一步即可运行完整流程 (`dvc repro`)

✅ **数据管理**
- 使用 DVC 追踪数据文件
- 数据预处理独立成阶段
- 支持训练/验证/测试集划分

✅ **实验追踪**
- 使用 DVC 实验功能 (`dvc exp`)
- 参数与指标关联
- 支持多实验对比

### L2-10 (指标追踪标准)

✅ **至少 10 个性能指标**
- 准确率 (Accuracy)
- 平衡准确率 (Balanced Accuracy)
- 精确率 (Precision) - 宏平均和加权平均
- 召回率 (Recall) - 宏平均和加权平均
- F1 分数 - 宏平均和加权平均
- ROC AUC (OvR/OvO)
- PR AUC
- Top-K 准确率
- Cohen's Kappa
- Matthews Correlation Coefficient
- Jaccard Score
- 对数损失 (Log Loss)

## 技术栈

- **深度学习框架**: PyTorch
- **数据版本控制**: DVC
- **实验管理**: DVC Experiments
- **模型格式**: torch.save() (.pt/.pth)
- **配置管理**: YAML (params.yaml)
- **数据加载**: Dataset + DataLoader

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 初始化 DVC

```bash
# 初始化 DVC（如果尚未初始化）
dvc init

# 配置远程存储（可选）
dvc remote add -d myremote s3://mybucket/dvc-store
# 或: dvc remote add -d myremote /path/to/local/storage
```

### 3. 运行流水线

```bash
# 运行完整流水线
dvc repro

# 或分阶段运行
dvc repro generate_data    # 数据生成
dvc repro preprocess       # 数据预处理
dvc repro train            # 模型训练
dvc repro evaluate         # 模型评估
```

### 4. 查看指标

```bash
# 查看指标
dvc metrics show

# 查看详细指标
cat metrics/metrics.json

# 查看评估报告
cat reports/evaluation_report.md
```

## 实验管理

### 运行新实验

```bash
# 修改 params.yaml 中的参数，然后运行
dvc exp run

# 或使用临时参数
dvc exp run --set-param train.learning_rate=0.01

# 使用不同的模型架构
dvc exp run --set-param model.hidden_dims=[256,128,64]
```

### 查看实验历史

```bash
# 查看所有实验
dvc exp show

# 以表格形式查看
dvc exp show --csv

# 对比实验
dvc exp diff exp-1 exp-2
```

### 保存和恢复实验

```bash
# 保存实验到 Git 分支
dvc exp branch exp-abc123 experiment-1

# 应用实验到工作区
dvc exp apply exp-abc123

# 清理临时实验
dvc exp gc
```

## 配置说明

### params.yaml

所有超参数集中管理：

```yaml
# 数据参数
data:
  n_samples: 5000
  n_features: 20
  n_classes: 3
  test_size: 0.2
  val_size: 0.1

# 模型参数
model:
  input_dim: 20
  hidden_dims: [128, 64, 32]
  output_dim: 3
  dropout_rate: 0.3

# 训练参数
train:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"

# 早停参数
early_stopping:
  enabled: true
  patience: 10
```

## 流水线阶段

### 1. 数据生成 (`generate_data`)

使用 sklearn 生成合成分类数据集。

**输入**: `params.yaml`

**输出**:
- `data/X_raw.npy`
- `data/y_raw.npy`
- `data/data_info.json`

### 2. 数据预处理 (`preprocess`)

数据标准化、训练/验证/测试集划分。

**输入**:
- `data/X_raw.npy`
- `data/y_raw.npy`

**输出**:
- `data/X_train.npy`, `data/y_train.npy`
- `data/X_val.npy`, `data/y_val.npy`
- `data/X_test.npy`, `data/y_test.npy`

### 3. 模型训练 (`train`)

使用 PyTorch 训练 MLP 分类器。

**特性**:
- 手动训练循环
- 学习率调度
- 早停机制
- 模型检查点保存

**输入**:
- 预处理后的数据
- `params.yaml`

**输出**:
- `models/model.pt`
- `models/checkpoint_best.pt`
- `metrics/training_history.json`

### 4. 模型评估 (`evaluate`)

计算 12+ 个性能指标，生成评估报告。

**输入**:
- 测试数据
- 训练好的模型

**输出**:
- `metrics/metrics.json`
- `metrics/evaluation_metrics.json`
- `reports/evaluation_report.md`

## 模型架构

### MLPClassifier

```python
MLPClassifier(
    input_dim=20,
    hidden_dims=[128, 64, 32],
    output_dim=3,
    dropout_rate=0.3,
    activation="relu"
)
```

架构：
```
Input (20) → Linear(20, 128) → ReLU → Dropout(0.3)
           → Linear(128, 64) → ReLU → Dropout(0.3)
           → Linear(64, 32) → ReLU → Dropout(0.3)
           → Linear(32, 3) → Output
```

## 训练策略

### 优化器
- Adam (默认)
- SGD (可选)
- AdamW (可选)

### 学习率调度
- ReduceLROnPlateau (默认)
- StepLR
- ExponentialLR

### 早停
- 监控指标: `val_loss` 或 `val_accuracy`
- 耐心值: 10 epochs
- 最小变化: 0.0001

## 数据版本控制

### 追踪数据

```bash
# 添加数据到 DVC
dvc add data/X_raw.npy data/y_raw.npy

# 提交到 Git
git add data/*.dvc .gitignore
git commit -m "Add raw data"

# 推送到远程存储
dvc push
```

### 拉取数据

```bash
# 从远程拉取数据
dvc pull

# 或只拉取特定文件
dvc pull data/X_raw.npy
```

## 常见问题

### Q: 如何修改超参数？

A: 编辑 `params.yaml` 文件，然后运行 `dvc repro`。

### Q: 如何只运行特定阶段？

A: 使用 `dvc repro <stage_name>`，例如 `dvc repro train`。

### Q: 如何查看训练进度？

A: 训练脚本会输出进度条和指标到控制台。

### Q: 如何导出模型？

A: 训练好的模型保存在 `models/model.pt`，可以直接加载使用。

### Q: 如何添加新的评估指标？

A: 编辑 `src/evaluate.py` 中的 `calculate_metrics` 函数。

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

MIT License

## 相关资源

- [DVC 文档](https://dvc.org/doc)
- [PyTorch 文档](https://pytorch.org/docs/)
- [DVC + PyTorch 教程](https://dvc.org/doc/use-cases/versioning-data-and-models)
