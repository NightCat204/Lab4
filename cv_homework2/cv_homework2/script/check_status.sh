#!/bin/bash
# Quick check training status

cd /home/ba3033/ws/resnet/Lab4/cv_homework2/cv_homework2

echo "=========================================="
echo "Training Status Check (AdamW Optimizer)"
echo "=========================================="

if [ -f "report/resnet50_adamw_best.pth" ]; then
    echo "✓ ResNet50 training: COMPLETED"
    echo "  Checkpoint: report/resnet50_adamw_best.pth"
    
    if [ -f "report/distilled_resnet18_adamw_best.pth" ]; then
        echo "✓ Knowledge Distillation: COMPLETED"
        echo "  Student checkpoint: report/distilled_resnet18_adamw_best.pth"
        echo ""
        echo "All tasks finished! Check report/ for results."
    else
        echo "⧗ Knowledge Distillation: NOT STARTED"
        echo ""
        echo "Run: bash script/run_knowledge_distillation.sh"
    fi
else
    echo "⧗ ResNet50 training: IN PROGRESS"
    echo "  Expected time: ~50 minutes"
    echo ""
    echo "Training will automatically save checkpoint when complete."
fi

echo "=========================================="
