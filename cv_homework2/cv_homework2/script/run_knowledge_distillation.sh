#!/bin/bash
# Knowledge Distillation: AdamW-trained ResNet50 -> ResNet18
# Run this after ResNet50 teacher model training is complete

cd /home/ba3033/ws/resnet/Lab4/cv_homework2/cv_homework2
source /home/ba3033/miniconda3/bin/activate resnet

echo "========================================"
echo "Knowledge Distillation Experiment"
echo "Teacher: ResNet50 (AdamW) -> Student: ResNet18"
echo "========================================"

# Check if teacher checkpoint exists
TEACHER_CHECKPOINT="report/resnet50_adamw_best.pth"
if [ ! -f "$TEACHER_CHECKPOINT" ]; then
    echo "Error: Teacher checkpoint not found at $TEACHER_CHECKPOINT"
    echo "Please train ResNet50 first using resnet-depth-comparison.py"
    exit 1
fi

echo -e "\nTeacher checkpoint found: $TEACHER_CHECKPOINT"

# Run knowledge distillation
echo -e "\n=========================================="
echo "Configuration: T=4.0, α=0.3, β=0.3, AdamW optimizer"
echo "=========================================="
python src/knowledge_distillation.py \
    --teacher-checkpoint $TEACHER_CHECKPOINT \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.001 \
    --temperature 4.0 \
    --alpha 0.3 \
    --beta 0.3 \
    --device 0 \
    --output report/distilled_resnet18_adamw

echo -e "\n=========================================="
echo "Generating comparison visualization..."
echo "=========================================="
python src/plot_distillation_comparison.py \
    --distilled report/distilled_resnet18_adamw_results.json \
    --baseline report/resnet18_result.json \
    --teacher report/resnet50_adamw.json \
    --output report/distillation_comparison_adamw.png

echo -e "\n=========================================="
echo "Knowledge Distillation Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - Model: report/distilled_resnet18_adamw_best.pth"
echo "  - Metrics: report/distilled_resnet18_adamw_results.json"
echo "  - Visualization: report/distillation_comparison_adamw.png"
