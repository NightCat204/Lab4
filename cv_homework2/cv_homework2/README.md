# ResNet9 on CIFAR-10

pytorch版本最好是1.10

**现已支持多GPU并行训练！** 🚀

## Project Structure

```
cv_homework2/
├── src/                    # Source code
│   ├── resnet-cifar100.py              # Original SGD version
│   ├── resnet-cifar100-adam.py         # Adam optimizer version
│   ├── resnet-cifar100-adamw.py        # AdamW optimizer version
│   ├── resnet-cifar100-lr-analysis.py  # LR analysis with scheduler
│   └── plot_lr_comparison.py           # Visualization script
├── config/                 # Configuration files
│   ├── config.yml
│   ├── config_adam.yml
│   ├── config_adamw.yml
│   └── config_multi_gpu.yml
├── script/                 # Shell scripts
│   ├── run_lr_experiments.sh           # Batch LR experiments
│   └── train_dual_gpu.sh               # Dual-GPU training
├── data/                   # Dataset directory
├── report/                 # Training results and plots
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### 单GPU训练

Run the example on the CPU:

```bash
python src/resnet-cifar100.py -c config/config.yml
```

Run the example on the GPU (device 0):

```bash
# SGD optimizer (original)
python src/resnet-cifar100.py -c config/config.yml --device 0

# Adam optimizer
python src/resnet-cifar100-adam.py -c config/config_adam.yml --device 0

# AdamW optimizer
python src/resnet-cifar100-adamw.py -c config/config_adamw.yml --device 0
```

### 双GPU并行训练（推荐）

使用两块3090同时训练，速度提升约1.5-2倍：

```bash
# 方式1: 直接运行
python src/resnet-cifar100.py -c config/config_multi_gpu.yml --device 0,1

# 方式2: 使用快速启动脚本
./script/train_dual_gpu.sh
```

### 学习率分析实验

```bash
# 单次实验（固定学习率）
python src/resnet-cifar100-lr-analysis.py --lr 0.1 --epochs 50 --device 0 --output-name report/test

# 单次实验（动态学习率：epoch 30和40降低10倍）
python src/resnet-cifar100-lr-analysis.py --lr 0.1 --epochs 50 --device 0 --lr-schedule --output-name report/test_scheduled

# 批量运行所有LR对比实验
./script/run_lr_experiments.sh
```

### 更多选项

```bash
# 使用所有可用GPU
python src/resnet-cifar100.py -c config/config_multi_gpu.yml --device all

# 后台运行（推荐用于长时间训练）
nohup python src/resnet-cifar100.py -c config/config_multi_gpu.yml --device 0,1 > training.log 2>&1 &
```

## 多GPU训练详细说明

详见 [MULTI_GPU_USAGE.md](MULTI_GPU_USAGE.md)，包括：
- 使用方法和参数说明
- 性能对比数据
- 配置文件说明
- 故障排除指南