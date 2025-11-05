# PosFormer

<h3 align="center"> <a href="https://arxiv.org/abs/2407.07764">PosFormer: 基于位置森林 Transformer 的复杂手写数学表达式识别</a></h3>


<h5 align="center">


# 描述
本仓库提供了手写数学表达式识别（HMER）模型 **位置森林 Transformer (PosFormer)** 的实现。这一创新模型引入了一种双任务方法，同时优化表达式识别和位置识别，从而促进了数学表达式中符号的位置感知特征学习。它采用一种名为“位置森林”的新颖结构来解析和建模符号的层次关系与空间定位，且无需额外的标注。此外，一个隐式注意力修正模块被集成到基于序列的解码器架构中，以增强符号识别的专注度和准确性。PosFormer 在多个基准测试（包括单行的 CROHME 数据集以及更复杂的多行和嵌套表达式数据集）上均表现出显著优于现有方法的性能，在不增加额外计算开销的情况下实现了更高的性能。本仓库包含代码、预训练模型和使用说明，以帮助研究人员和开发者应用并进一步开发这一先进的 HMER 解决方案。


## Getting Started

### Installation
```bash
cd PosFormer-main

# install project   
conda env create -n PosFormer -f environment.yml
```
### 数据准备
数据集结构如下:
```
📂 data
   |
   └── 📂 train
   |
   └── 📂 val
       |   ├── 📂 img
       │  ├── 0.png
       │  ├── 1.png
       │  └── ...
       └── caption.txt
   |
   └── 📂 test
       ├── 📂 img
       │   ├── 0.png
       │   ├── 1.png
       │   └── ...
       └── caption.txt
```


### 开始训练
```bash
cd PosFormer-main
python train.py --config config.yaml
```

### 模型评估
```bash
cd PosFormer
bash eval_all_crohme.sh ./path/to/checkpoints
# 注意：推理过程只能够使用单卡进行
```

### 查看实验结果 
```
tensorboard  --logdir./path/to/lightning_logs/version_deeplearning_data_1 --port 6009
```