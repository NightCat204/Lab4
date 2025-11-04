#!/bin/bash
# ResNet depth comparison experiments

cd /home/ba3033/ws/resnet/Lab4/cv_homework2/cv_homework2
source /home/ba3033/miniconda3/bin/activate resnet

echo "========================================"
echo "ResNet Depth Comparison Experiments"
echo "ResNet10 vs ResNet18 vs ResNet50"
echo "========================================"
echo ""

DEVICE="0"
EPOCHS=50
BATCH_SIZE=256
LR=0.1

# Experiment 1: ResNet10
echo "[1/3] Training ResNet10..."
python src/resnet-depth-comparison.py \
    --model resnet10 \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --lr $LR \
    --device $DEVICE \
    --output-name report/resnet10_result

echo ""

# Experiment 2: ResNet18
echo "[2/3] Training ResNet18..."
python src/resnet-depth-comparison.py \
    --model resnet18 \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --lr $LR \
    --device $DEVICE \
    --output-name report/resnet18_result

echo ""

# Experiment 3: ResNet50
echo "[3/3] Training ResNet50..."
python src/resnet-depth-comparison.py \
    --model resnet50 \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --lr $LR \
    --device $DEVICE \
    --output-name report/resnet50_result

echo ""
echo "========================================"
echo "Generating comparison visualization..."
echo "========================================"
python src/plot_depth_comparison.py \
    report/resnet10_result.json \
    report/resnet18_result.json \
    report/resnet50_result.json

echo ""
echo "All experiments completed!"
echo "Results saved in report/ directory"
