#!/bin/bash
# 双GPU训练快速启动脚本

echo "========================================"
echo "ResNet CIFAR-100 双GPU训练"
echo "========================================"
echo ""

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
echo ""

# 询问是否继续
read -p "是否使用GPU 0和1进行训练? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "开始双GPU训练..."
    echo "日志将保存到 training_dual_gpu.log"
    echo "使用 'tail -f training_dual_gpu.log' 查看进度"
    echo ""
    
    # 后台运行
    nohup python resnet-cifar100.py -c config_multi_gpu.yml --device 0,1 > training_dual_gpu.log 2>&1 &
    
    PID=$!
    echo "训练进程已启动，PID: $PID"
    echo ""
    echo "有用的命令："
    echo "  查看日志: tail -f training_dual_gpu.log"
    echo "  监控GPU: watch -n 1 nvidia-smi"
    echo "  停止训练: kill $PID"
    echo ""
    
    # 显示前几行输出
    sleep 3
    echo "初始输出："
    head -20 training_dual_gpu.log
else
    echo "已取消"
fi
