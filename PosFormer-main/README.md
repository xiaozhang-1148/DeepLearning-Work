# PosFormer

<h3 align="center"> <a href="https://arxiv.org/abs/2407.07764">PosFormer: 基于位置森林 Transformer 的复杂手写数学表达式识别</a></h3>


<h5 align="center">


# 描述
本仓库提供了手写数学表达式识别（HMER）模型 **位置森林 Transformer (PosFormer)** 的实验复现。这一模型创新性的引入了一种双任务方法，同时优化表达式识别和位置识别，从而促进了数学表达式中符号的位置感知特征学习。它采用一种名为“位置森林”的新颖结构来解析和建模符号的层次关系与空间定位，且无需额外的标注。此外，一个隐式注意力修正模块被集成到基于序列的解码器架构中，以增强符号识别的专注度和准确性。


## 项目结构

本项目已重构为标准的 Python 项目结构：

```
PosFormer-main/
├── 📁 src/                          # 源代码目录
│   └── posformer/                  # 核心包
│       ├── core/                   # 核心模块 (模型, 处理器)
│       ├── datamodule/             # 数据加载模块
│       └── utils/                  # 工具模块
├── 📁 configs/                     # 配置文件
├── 📁 scripts/                     # 训练和评估脚本
├── 📁 data/                        # 数据目录
│   └── raw/                        # 原始数据 (zip文件等)
├── 📁 tests/                       # 测试代码
├── 📄 Makefile                    # 常用命令封装
├── 📄 setup.py                    # 安装配置
└── 📄 requirements.txt             # 依赖列表
```

## 快速开始 (Getting Started)

### 安装 (Installation)

1. **环境配置**
   推荐使用 Conda 创建环境：
   ```bash
   conda env create -n PosFormer -f environment.yml
   conda activate PosFormer
   ```

2. **安装项目依赖**
   以开发模式安装本项目，以便在任何地方调用 `posformer` 包：
   ```bash
   pip install -e .
   # 或者使用 Makefile
   make install
   ```

### 数据准备 (Data Preparation)
请将数据集文件（如 `.zip`）放置在 `data/raw/` 目录下。
解压后的标准结构应如下所示（脚本会自动处理 zip 文件）：
```
📂 data/raw
   ├── deeplearning_dataset.zip
   ├── ours_Dataset.zip
   └── ...
```

### 训练 (Training)

使用默认配置开始训练：
```bash
# 使用 Makefile
make train

# 或者直接运行脚本
python scripts/train.py --config configs/config_deep_data.yaml
```

### 评估 (Evaluation)

对模型进行评估：
```bash
# 使用 Makefile (运行默认评估脚本)
make evaluate

# 或者手动运行评估脚本
# 注意：推理过程建议使用单卡
python scripts/evaluate.py --path lightning_logs/version_xxx/checkpoints/ --dataset-zip data/raw/deeplearning_dataset.zip
```

或者使用提供的 shell 脚本批量评估：
```bash
bash scripts/eval_all_deep_data.sh lightning_logs/version_xxx/checkpoints/
```

### 查看实验结果 (Visualization)
使用 TensorBoard 查看训练日志：
```bash
tensorboard --logdir lightning_logs/ --port 6008
```

## 开发指南

- **代码风格检查**: 提交前请运行 `pre-commit run --all-files`。
- **运行测试**: 使用 `make test` 运行单元测试。