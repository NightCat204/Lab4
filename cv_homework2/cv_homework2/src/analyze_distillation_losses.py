import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def analyze_distillation_losses(results_file, output_file):
    """
    Analyze and visualize knowledge distillation loss components
    Focus on soft target loss and logits loss contributions
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    config = data['config']
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Extract loss components
    total_loss = history['train_loss']
    hard_loss = history['train_hard_loss']
    soft_loss = history['train_soft_loss']
    logits_loss = history['train_logits_loss']
    test_acc = history['test_accuracy']
    
    # Calculate weighted contributions
    alpha = config['alpha']
    beta = config['beta']
    gamma = 1 - alpha - beta
    
    soft_contribution = [alpha * s for s in soft_loss]
    logits_contribution = [beta * l for l in logits_loss]
    hard_contribution = [gamma * h for h in hard_loss]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # ========== Plot 1: Loss Components Over Time ==========
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, soft_loss, 'c-', linewidth=2.5, label=f'Soft Target Loss (KD)', alpha=0.8)
    ax1.plot(epochs, logits_loss, 'b-', linewidth=2.5, label=f'Logits Loss (MSE)', alpha=0.8)
    ax1.plot(epochs, hard_loss, 'r--', linewidth=1.5, label=f'Hard Loss (CE)', alpha=0.6)
    ax1.axvline(x=30, color='gray', linestyle=':', alpha=0.5, label='LR Decay (0.1x)')
    ax1.axvline(x=40, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Loss Value', fontsize=13)
    ax1.set_title('Knowledge Distillation Loss Components', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # ========== Plot 2: Weighted Loss Contributions ==========
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, soft_contribution, 'c-', linewidth=2, label=f'α·Soft (α={alpha})')
    ax2.plot(epochs, logits_contribution, 'b-', linewidth=2, label=f'β·Logits (β={beta})')
    ax2.plot(epochs, hard_contribution, 'r-', linewidth=2, label=f'γ·Hard (γ={gamma})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Weighted Loss', fontsize=12)
    ax2.set_title('Weighted Contributions to Total Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ========== Plot 3: Stacked Area Chart ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.stackplot(epochs, hard_contribution, soft_contribution, logits_contribution,
                  labels=[f'Hard ({gamma:.1f})', f'Soft ({alpha:.1f})', f'Logits ({beta:.1f})'],
                  colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Stacked Loss', fontsize=12)
    ax3.set_title('Loss Components Stacked', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========== Plot 4: Loss Ratios ==========
    ax4 = fig.add_subplot(gs[1, 1])
    soft_ratio = [s / (s + l + h) for s, l, h in zip(soft_loss, logits_loss, hard_loss)]
    logits_ratio = [l / (s + l + h) for s, l, h in zip(soft_loss, logits_loss, hard_loss)]
    hard_ratio = [h / (s + l + h) for s, l, h in zip(soft_loss, logits_loss, hard_loss)]
    
    ax4.plot(epochs, soft_ratio, 'c-', linewidth=2, label='Soft Loss Ratio')
    ax4.plot(epochs, logits_ratio, 'b-', linewidth=2, label='Logits Loss Ratio')
    ax4.plot(epochs, hard_ratio, 'r-', linewidth=2, label='Hard Loss Ratio')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Ratio', fontsize=12)
    ax4.set_title('Relative Loss Proportions', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # ========== Plot 5: Loss Correlation with Accuracy ==========
    ax5 = fig.add_subplot(gs[1, 2])
    ax5_twin = ax5.twinx()
    
    ln1 = ax5.plot(epochs, soft_loss, 'c-', linewidth=2, label='Soft Loss')
    ln2 = ax5.plot(epochs, logits_loss, 'b-', linewidth=2, label='Logits Loss')
    ln3 = ax5_twin.plot(epochs, test_acc, 'g-', linewidth=2.5, label='Test Accuracy', alpha=0.7)
    
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss Value', fontsize=12, color='blue')
    ax5_twin.set_ylabel('Test Accuracy (%)', fontsize=12, color='green')
    ax5.set_title('Loss vs Accuracy', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='y', labelcolor='blue')
    ax5_twin.tick_params(axis='y', labelcolor='green')
    
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ========== Plot 6: Loss Reduction Rate ==========
    ax6 = fig.add_subplot(gs[2, 0])
    soft_reduction = [0] + [soft_loss[i-1] - soft_loss[i] for i in range(1, len(soft_loss))]
    logits_reduction = [0] + [logits_loss[i-1] - logits_loss[i] for i in range(1, len(logits_loss))]
    
    ax6.plot(epochs, soft_reduction, 'c-', linewidth=2, label='Soft Loss Reduction', alpha=0.7)
    ax6.plot(epochs, logits_reduction, 'b-', linewidth=2, label='Logits Loss Reduction', alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Loss Reduction', fontsize=12)
    ax6.set_title('Loss Reduction Rate per Epoch', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # ========== Plot 7: Final Epoch Loss Breakdown ==========
    ax7 = fig.add_subplot(gs[2, 1])
    final_losses = [soft_loss[-1], logits_loss[-1], hard_loss[-1]]
    final_contributions = [soft_contribution[-1], logits_contribution[-1], hard_contribution[-1]]
    
    x = np.arange(3)
    width = 0.35
    bars1 = ax7.bar(x - width/2, final_losses, width, label='Raw Loss', 
                    color=['cyan', 'blue', 'red'], alpha=0.6)
    bars2 = ax7.bar(x + width/2, final_contributions, width, label='Weighted Contribution',
                    color=['cyan', 'blue', 'red'], alpha=0.9)
    
    ax7.set_ylabel('Loss Value', fontsize=12)
    ax7.set_title(f'Final Epoch Loss Analysis (Epoch {len(epochs)})', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(['Soft\n(KD)', 'Logits\n(MSE)', 'Hard\n(CE)'])
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ========== Plot 8: Statistics Table ==========
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('tight')
    ax8.axis('off')
    
    # Calculate statistics
    soft_initial = soft_loss[0]
    soft_final = soft_loss[-1]
    soft_reduction_pct = (soft_initial - soft_final) / soft_initial * 100
    
    logits_initial = logits_loss[0]
    logits_final = logits_loss[-1]
    logits_reduction_pct = (logits_initial - logits_final) / logits_initial * 100
    
    table_data = [
        ['Metric', 'Soft Target', 'Logits', 'Hard'],
        ['Initial Loss', f'{soft_initial:.3f}', f'{logits_initial:.3f}', f'{hard_loss[0]:.3f}'],
        ['Final Loss', f'{soft_final:.3f}', f'{logits_final:.3f}', f'{hard_loss[-1]:.3f}'],
        ['Reduction %', f'{soft_reduction_pct:.1f}%', f'{logits_reduction_pct:.1f}%', 
         f'{(hard_loss[0]-hard_loss[-1])/hard_loss[0]*100:.1f}%'],
        ['Weight (α/β/γ)', f'{alpha}', f'{beta}', f'{gamma}'],
        ['Final Contrib.', f'{soft_contribution[-1]:.3f}', 
         f'{logits_contribution[-1]:.3f}', f'{hard_contribution[-1]:.3f}'],
        ['Avg Loss', f'{np.mean(soft_loss):.3f}', f'{np.mean(logits_loss):.3f}', 
         f'{np.mean(hard_loss):.3f}'],
    ]
    
    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    colors = ['#E7E6E6', '#F2F2F2']
    for i in range(1, len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i % 2])
    
    ax8.set_title('Loss Components Statistics', fontsize=12, fontweight='bold', pad=15)
    
    # Main title
    fig.suptitle(f'Knowledge Distillation Loss Analysis\n'
                 f'Teacher: ResNet50 ({data["teacher_accuracy"]:.2f}%) → '
                 f'Student: ResNet18 ({data["student_best_accuracy"]:.2f}%)',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # Print analysis
    print(f"\n{'='*70}")
    print(f"KNOWLEDGE DISTILLATION LOSS ANALYSIS")
    print(f"{'='*70}")
    print(f"Configuration: T={config['temperature']}, α={alpha}, β={beta}, γ={gamma}")
    print(f"\nSoft Target Loss (KD):")
    print(f"  Initial: {soft_initial:.4f} → Final: {soft_final:.4f}")
    print(f"  Reduction: {soft_reduction_pct:.1f}%")
    print(f"  Average: {np.mean(soft_loss):.4f}")
    print(f"\nLogits Loss (MSE):")
    print(f"  Initial: {logits_initial:.4f} → Final: {logits_final:.4f}")
    print(f"  Reduction: {logits_reduction_pct:.1f}%")
    print(f"  Average: {np.mean(logits_loss):.4f}")
    print(f"\nStudent Performance:")
    print(f"  Best Accuracy: {data['student_best_accuracy']:.2f}%")
    print(f"  Teacher Accuracy: {data['teacher_accuracy']:.2f}%")
    print(f"  Gap: {data['teacher_accuracy'] - data['student_best_accuracy']:.2f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Knowledge Distillation Losses')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to distillation results JSON file')
    parser.add_argument('--output', type=str, default='report/distillation_loss_analysis.png',
                        help='Output file path for visualization')
    args = parser.parse_args()
    
    analyze_distillation_losses(args.results, args.output)
