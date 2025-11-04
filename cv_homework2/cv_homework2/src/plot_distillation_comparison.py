import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_results(file_path):
    """Load training results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_distillation_comparison(distilled_results, baseline_file, teacher_file, output_file):
    """
    Visualize knowledge distillation results comparing:
    - Distilled ResNet18 (student trained with KD)
    - Baseline ResNet18 (student trained normally)
    - ResNet50 (teacher)
    """
    # Load baseline and teacher results
    baseline_results = load_results(baseline_file)
    teacher_results = load_results(teacher_file)
    
    # Extract data
    epochs = range(1, len(distilled_results['history']['test_accuracy']) + 1)
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========== Plot 1: Test Accuracy Comparison ==========
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, distilled_results['history']['test_accuracy'], 
             'b-', linewidth=2, label='Distilled ResNet18 (Student)')
    ax1.plot(epochs, baseline_results['epoch_accuracies'], 
             'g--', linewidth=2, label='Baseline ResNet18')
    ax1.plot(epochs, teacher_results['epoch_accuracies'], 
             'r-.', linewidth=2, label='ResNet50 (Teacher)', alpha=0.7)
    ax1.axhline(y=distilled_results['student_best_accuracy'], 
                color='b', linestyle=':', alpha=0.5, label=f'Best Distilled: {distilled_results["student_best_accuracy"]:.2f}%')
    ax1.axhline(y=baseline_results['best_accuracy'], 
                color='g', linestyle=':', alpha=0.5, label=f'Best Baseline: {baseline_results["best_accuracy"]:.2f}%')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy Comparison: Distillation vs Baseline vs Teacher', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== Plot 2: Loss Components Breakdown ==========
    ax2 = fig.add_subplot(gs[0, 2])
    loss_components = [
        distilled_results['history']['train_hard_loss'][-1],
        distilled_results['history']['train_soft_loss'][-1],
        distilled_results['history']['train_logits_loss'][-1]
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax2.bar(['Hard Loss\n(CE)', 'Soft Loss\n(KD)', 'Logits Loss\n(MSE)'], 
            loss_components, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Loss Value (Final Epoch)', fontsize=11)
    ax2.set_title('Loss Components Breakdown', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(loss_components):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # ========== Plot 3: Training Loss Comparison ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, distilled_results['history']['train_loss'], 
             'b-', linewidth=2, label='Distilled (Total Loss)')
    ax3.plot(epochs, baseline_results['epoch_losses'], 
             'g--', linewidth=2, label='Baseline (CE Loss)')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Loss', fontsize=12)
    ax3.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # ========== Plot 4: Individual Loss Components Over Time ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, distilled_results['history']['train_hard_loss'], 
             'r-', linewidth=1.5, label='Hard Loss (CE)', alpha=0.8)
    ax4.plot(epochs, distilled_results['history']['train_soft_loss'], 
             'c-', linewidth=1.5, label='Soft Loss (KD)', alpha=0.8)
    ax4.plot(epochs, distilled_results['history']['train_logits_loss'], 
             'b-', linewidth=1.5, label='Logits Loss (MSE)', alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Value', fontsize=12)
    ax4.set_title('Distillation Loss Components', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ========== Plot 5: Accuracy Gain Over Epochs ==========
    ax5 = fig.add_subplot(gs[1, 2])
    accuracy_gain = [d - b for d, b in zip(distilled_results['history']['test_accuracy'], 
                                             baseline_results['epoch_accuracies'])]
    colors_gain = ['green' if x >= 0 else 'red' for x in accuracy_gain]
    ax5.bar(epochs, accuracy_gain, color=colors_gain, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Accuracy Gain (%)', fontsize=12)
    ax5.set_title('Distilled vs Baseline Accuracy Gain', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ========== Plot 6: Summary Statistics Table ==========
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate improvements
    best_gain = distilled_results['student_best_accuracy'] - baseline_results['best_accuracy']
    final_gain = distilled_results['student_final_accuracy'] - baseline_results['final_accuracy']
    
    table_data = [
        ['Metric', 'Distilled ResNet18', 'Baseline ResNet18', 'ResNet50 Teacher', 'Improvement'],
        ['Best Test Accuracy', 
         f"{distilled_results['student_best_accuracy']:.2f}%",
         f"{baseline_results['best_accuracy']:.2f}%",
         f"{teacher_results['best_accuracy']:.2f}%",
         f"+{best_gain:.2f}%" if best_gain > 0 else f"{best_gain:.2f}%"],
        ['Final Test Accuracy',
         f"{distilled_results['student_final_accuracy']:.2f}%",
         f"{baseline_results['final_accuracy']:.2f}%",
         f"{teacher_results['final_accuracy']:.2f}%",
         f"+{final_gain:.2f}%" if final_gain > 0 else f"{final_gain:.2f}%"],
        ['Training Time',
         f"{distilled_results['training_time']/60:.1f} min",
         f"{baseline_results['training_time']/60:.1f} min",
         f"{teacher_results['training_time']/60:.1f} min",
         f"{(distilled_results['training_time']-baseline_results['training_time'])/60:+.1f} min"],
        ['Parameters',
         '11.22M',
         '11.22M',
         '23.71M',
         'Same'],
        ['Temperature',
         f"{distilled_results['config']['temperature']}",
         'N/A',
         'N/A',
         'N/A'],
        ['Loss Weights (α, β)',
         f"α={distilled_results['config']['alpha']}, β={distilled_results['config']['beta']}",
         'N/A',
         'N/A',
         'N/A'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
            
            # Highlight improvement column
            if j == 4 and i > 0:
                if '+' in table_data[i][j]:
                    cell.set_facecolor('#C6EFCE')
                elif table_data[i][j] != 'N/A' and table_data[i][j] != 'Same':
                    cell.set_facecolor('#FFC7CE')
    
    ax6.set_title('Knowledge Distillation Summary Statistics', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle('Knowledge Distillation: ResNet50 → ResNet18 on CIFAR-100', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"KNOWLEDGE DISTILLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Teacher Model: ResNet50 ({teacher_results['best_accuracy']:.2f}%)")
    print(f"Baseline Student: ResNet18 ({baseline_results['best_accuracy']:.2f}%)")
    print(f"Distilled Student: ResNet18 ({distilled_results['student_best_accuracy']:.2f}%)")
    print(f"\nAccuracy Improvement: {best_gain:+.2f}%")
    print(f"Temperature: {distilled_results['config']['temperature']}")
    print(f"Loss Weights: α={distilled_results['config']['alpha']}, β={distilled_results['config']['beta']}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Knowledge Distillation Results')
    parser.add_argument('--distilled', type=str, required=True,
                        help='Path to distilled model results JSON')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline ResNet18 results JSON')
    parser.add_argument('--teacher', type=str, required=True,
                        help='Path to teacher ResNet50 results JSON')
    parser.add_argument('--output', type=str, default='report/distillation_comparison.png',
                        help='Output file path for visualization')
    args = parser.parse_args()
    
    distilled_results = load_results(args.distilled)
    plot_distillation_comparison(distilled_results, args.baseline, args.teacher, args.output)
