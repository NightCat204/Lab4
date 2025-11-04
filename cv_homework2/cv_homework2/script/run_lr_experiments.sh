#!/bin/bash
# Batch training script for learning rate analysis

# Change to working directory
cd /home/ba3033/ws/resnet/Lab4/cv_homework2/cv_homework2

# Activate conda environment
source /home/ba3033/miniconda3/bin/activate resnet

echo "========================================"
echo "Learning Rate Analysis Experiments"
echo "========================================"
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Configuration
DEVICE="0"
EPOCHS=50
BATCH_SIZE=256

# Experiment 1: LR = 0.001 (Fixed)
echo "[1/4] Training with LR=0.001 (Fixed)..."
python src/resnet-cifar100-lr-analysis.py \
    --lr 0.001 \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --device $DEVICE \
    --output-name report/lr_0.001_fixed

# Experiment 2: LR = 0.01 (Fixed)
echo ""
echo "[2/4] Training with LR=0.01 (Fixed)..."
python src/resnet-cifar100-lr-analysis.py \
    --lr 0.01 \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --device $DEVICE \
    --output-name report/lr_0.01_fixed

# Experiment 3: LR = 0.1 (Fixed)
echo ""
echo "[3/4] Training with LR=0.1 (Fixed)..."
python src/resnet-cifar100-lr-analysis.py \
    --lr 0.1 \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --device $DEVICE \
    --output-name report/lr_0.1_fixed

# Experiment 4: LR = 0.1 (Scheduled: decay at epoch 30, 40)
echo ""
echo "[4/4] Training with LR=0.1 (Scheduled)..."
python src/resnet-cifar100-lr-analysis.py \
    --lr 0.1 \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --device $DEVICE \
    --lr-schedule \
    --output-name report/lr_0.1_scheduled

# Generate comparison plots
echo ""
echo "========================================"
echo "Generating comparison plots..."
echo "========================================"
python src/plot_lr_comparison.py \
    --fixed-0.001 report/lr_0.001_fixed.json \
    --fixed-0.01 report/lr_0.01_fixed.json \
    --fixed-0.1 report/lr_0.1_fixed.json \
    --scheduled-0.1 report/lr_0.1_scheduled.json

echo ""
echo "All experiments completed!"
echo "Check report/lr_comparison.png for visualization"
