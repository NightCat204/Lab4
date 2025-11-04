#!/bin/bash
# Monitor training progress and automatically run distillation when complete

cd /home/ba3033/ws/resnet/Lab4/cv_homework2/cv_homework2
source /home/ba3033/miniconda3/bin/activate resnet

CHECKPOINT="report/resnet50_enhanced_best.pth"
JSON_RESULT="report/resnet50_enhanced.json"

echo "=========================================="
echo "Waiting for ResNet50 training to complete..."
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

# Wait for checkpoint file to be created
while [ ! -f "$CHECKPOINT" ]; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Training in progress..."
    sleep 60
done

# Wait a bit more to ensure file is fully written
sleep 10

echo -e "\n=========================================="
echo "ResNet50 training completed!"
echo "Starting knowledge distillation..."
echo "=========================================="

bash script/run_knowledge_distillation.sh

echo -e "\n=========================================="
echo "All tasks completed!"
echo "=========================================="
