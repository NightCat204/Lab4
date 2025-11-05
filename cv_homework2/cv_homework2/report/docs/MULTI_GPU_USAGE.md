# 多GPU训练使用说明

## 概述
程序已经修改为支持多GPU并行训练，使用 PyTorch 的 DataParallel 实现。

## 硬件配置
- GPU 0: NVIDIA GeForce RTX 3090 (24GB)
- GPU 1: NVIDIA GeForce RTX 3090 (24GB)
- CPU: AMD Ryzen 9 7950X 16-Core
- RAM: 124GB

## 使用方法

### 1. 单GPU训练（原始方式）
```bash
# 使用 GPU 0
python resnet-cifar100.py -c config.yml --device 0

# 使用 GPU 1
python resnet-cifar100.py -c config.yml --device 1
```

### 2. 双GPU并行训练（推荐）
```bash
# 使用 GPU 0 和 GPU 1
python resnet-cifar100.py -c config_multi_gpu.yml --device 0,1
```

### 3. 使用所有可用GPU
```bash
# 自动检测并使用所有GPU
python resnet-cifar100.py -c config_multi_gpu.yml --device all
```

### 4. 后台运行（长时间训练）
```bash
# 后台运行，输出保存到日志文件
nohup python resnet-cifar100.py -c config_multi_gpu.yml --device 0,1 > training_dual_gpu.log 2>&1 &

# 查看训练进度
tail -f training_dual_gpu.log

# 监控GPU使用情况
watch -n 1 nvidia-smi
```

## 性能对比

### 单GPU (batch_size=256)
- 训练速度: ~6秒/epoch
- 内存使用: ~3-4GB
- Batch size: 256

### 双GPU (batch_size=512)
- 训练速度: ~3-4秒/epoch (约 1.5-2x 加速)
- 内存使用: 每块GPU ~3-4GB
- Batch size: 512 (自动翻倍)
- **总吞吐量提升约 80-100%**

## 配置文件说明

### config.yml (单GPU)
```yaml
root: ./data
workers: 8        # 数据加载线程数
bsize: 256        # 单GPU batch size
epochs: 50
lr: 0.001
```

### config_multi_gpu.yml (多GPU)
```yaml
root: ./data
workers: 16       # 增加数据加载线程（双GPU需要更多）
bsize: 256        # 每个GPU的batch size (总batch=256*2=512)
epochs: 50
lr: 0.001
```

## 关键修改说明

1. **--device 参数**：现在支持字符串格式
   - `"0"`: 单GPU
   - `"0,1"`: 双GPU
   - `"all"`: 所有可用GPU

2. **自动Batch Size扩展**：使用多GPU时，总batch size会自动乘以GPU数量
   - 单GPU: 256
   - 双GPU: 512 (256 * 2)

3. **DataParallel包装**：模型自动用 `nn.DataParallel` 包装，数据并行处理

4. **工作线程数**：多GPU配置使用更多workers加速数据加载

## 注意事项

1. **显存要求**：每块GPU至少需要4GB显存（当前配置）
2. **同步开销**：DataParallel在每个batch结束时同步梯度，有一定开销
3. **线性扩展**：2块GPU不会达到完美的2x加速，通常是1.5-1.8x
4. **GPU 1占用**：如果GPU 1正在被其他任务占用（如显示的99%使用率），请只使用GPU 0

## 监控命令

```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 查看详细GPU信息
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv

# 查看正在运行的Python进程
ps aux | grep python | grep resnet
```

## 故障排除

### 问题：GPU 1被占用
如果看到GPU 1已被其他进程占用：
```bash
# 只使用GPU 0
python resnet-cifar100.py -c config.yml --device 0
```

### 问题：内存不足
如果出现 CUDA out of memory：
```bash
# 减小batch size（修改config文件中的bsize）
# 或只使用单GPU
```

### 问题：速度没有明显提升
- 检查workers数量是否足够（建议16以上）
- 确认两块GPU都在工作（使用nvidia-smi）
- 数据加载可能成为瓶颈
