#!/bin/bash

# 检查是否提供了模型版本号
if [ -z "$1" ]; then
    echo "错误: 请提供模型版本号作为第一个参数。"
    echo "用法: bash eval_all_deep_data.sh <version_name> [num_gpus]"
    exit 1
fi

# 从命令行参数获取版本号和 GPU 数量
VERSION=$1
# 如果没有提供第二个参数（GPU数量），则默认为 1
GPUS=${2:-1}

echo "--- 开始评估模型版本: $VERSION ---"
echo "--- 使用 GPU 数量: $GPUS ---"

# 遍历 'val' 和 'test' 数据集进行评估
for DATA_SPLIT in 'test'
do
    echo 
    echo "**************** 开始评估数据集: $DATA_SPLIT ****************"
    
    # --- 修改核心: 直接调用我们修改好的 test.py 脚本 ---
    python scripts/test/test.py \
        --path  "$VERSION" \
        --data-split "$DATA_SPLIT" \
        --gpus "$GPUS"
        
    echo "**************** 数据集评估完成: $DATA_SPLIT ****************"
done

echo
echo "--- 所有评估已完成 ---"