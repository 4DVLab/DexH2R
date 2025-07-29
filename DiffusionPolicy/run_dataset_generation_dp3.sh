#!/bin/bash

# 数据集生成脚本
# 按顺序运行三个命令来生成完整的数据集

echo "=== 开始数据集生成流程 ==="
echo "当前时间: $(date)"
echo ""

# 检查是否在正确的目录
if [ ! -f "dataset_generation/merge_dataset.py" ]; then
    echo "错误: 请在DiffusionPolicy目录下运行此脚本"
    exit 1
fi

# 检查grasping_period参数
GRASPING_PERIOD=""
if [ "$1" = "--grasping_period" ]; then
    GRASPING_PERIOD="--grasping_period"
    echo "使用grasping_period模式"
else
    echo "使用默认模式（不包含grasping_period）"
fi

echo ""

# 步骤1: 合并数据集
echo "步骤1: 运行 merge_dataset.py..."
python ./dataset_generation/merge_dataset.py $GRASPING_PERIOD
if [ $? -ne 0 ]; then
    echo "错误: merge_dataset.py 执行失败"
    exit 1
fi
echo "✓ merge_dataset.py 完成"
echo ""

# 步骤2: 合并图像
echo "步骤2: 运行 merge_images.py..."
python ./dataset_generation/merge_images.py $GRASPING_PERIOD
if [ $? -ne 0 ]; then
    echo "错误: merge_images.py 执行失败"
    exit 1
fi
echo "✓ merge_images.py 完成"
echo ""

# 步骤3: 合并zarr
echo "步骤3: 运行 merge_zarr.py..."
python merge_zarr.py --algo dp3
if [ $? -ne 0 ]; then
    echo "错误: merge_zarr.py 执行失败"
    exit 1
fi
echo "✓ merge_zarr.py 完成"
echo ""

echo "=== 数据集生成流程完成 ==="
echo "完成时间: $(date)"
echo "所有步骤已成功执行！" 