#!/bin/bash

if [ -z "$1" ]; then
    echo "错误: 请提供模型检查点文件(.ckpt)或其所在目录的路径。"
    echo "用法: bash eval_all_deep_data.sh <path_to_ckpt_or_dir> [num_gpus]"
    exit 1
fi

# 从命令行参数获取路径和 GPU 数量（第二个参数默认1）
INPUT_PATH=$1
GPUS=${2:-1}

# 如果传入的是一个文件，则取其所在的目录作为 --path 的值
# 否则，直接使用传入的目录路径
if [ -f "$INPUT_PATH" ]; then
    CKPT_DIR=$(dirname "$INPUT_PATH")
else
    CKPT_DIR=$INPUT_PATH
fi

echo "--- 开始评估模型，检查点目录: $CKPT_DIR ---"
echo "--- 使用 GPU 数量: $GPUS ---"

# 遍历需要评估的数据集（可根据需求扩展为 'test/easy' 'test/medium' 'test/hard'）
for DATA_SPLIT in 'test'
do
    echo 
    echo "**************** 开始评估数据集: $DATA_SPLIT ****************"
    
    # 调用 test.py，参数保持正确映射
    python scripts/test/test.py \
        --path "$CKPT_DIR" \
        --data-split "$DATA_SPLIT" \
        --gpus "$GPUS" \
        --dataset-zip "deeplearning_dataset.zip"
    
    echo "**************** 数据集评估完成: $DATA_SPLIT ****************"
done

echo
echo "--- 所有评估已完成 ---"