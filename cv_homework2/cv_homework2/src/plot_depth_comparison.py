import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def plot_depth_comparison(result_files, output_name='report/resnet_depth_comparison'):
    """Compare ResNet10/18/50 training dynamics"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # Load all results
    all_data = []
    for f in result_files:
        if os.path.exists(f):
            all_data.append(load_results(f))
    
    if not all_data:
        print("No result files found!")
        return
    
    # Main plot: Training Loss
    ax1 = fig.add_subplot(gs[0, :2])
    for idx, data in enumerate(all_data):
        epochs = range(1, len(data['epoch_losses']) + 1)
        ax1.plot(epochs, data['epoch_losses'], label=data['model'].upper(), 
                color=colors[idx], linewidth=2, marker=markers[idx], markersize=4, markevery=5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Test Accuracy
    ax2 = fig.add_subplot(gs[1, :2])
    for idx, data in enumerate(all_data):
        epochs = range(1, len(data['epoch_accuracies']) + 1)
        ax2.plot(epochs, [acc*100 for acc in data['epoch_accuracies']], 
                label=data['model'].upper(), color=colors[idx], linewidth=2, 
                marker=markers[idx], markersize=4, markevery=5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Loss (Log scale)
    ax3 = fig.add_subplot(gs[2, :2])
    for idx, data in enumerate(all_data):
        epochs = range(1, len(data['epoch_losses']) + 1)
        ax3.semilogy(epochs, data['epoch_losses'], label=data['model'].upper(),
                    color=colors[idx], linewidth=2, marker=markers[idx], markersize=4, markevery=5)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Training Loss (log scale)', fontsize=12, fontweight='bold')
    ax3.set_title('Training Loss (Log Scale)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, which='both')
    
    # Statistics table
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.axis('off')
    
    stats_data = []
    for data in all_data:
        stats_data.append([
            data['model'].upper(),
            f"{data['config']['total_params']/1e6:.2f}M",
            f"{data['best_accuracy']*100:.2f}%",
            f"{data['final_accuracy']*100:.2f}%"
        ])
    
    table = ax4.table(cellText=stats_data,
                     colLabels=['Model', 'Params', 'Best Acc', 'Final Acc'],
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(stats_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(['#E7E6E6', '#F2F2F2'][i % 2])
    
    ax4.set_title('Model Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Training time comparison
    ax5 = fig.add_subplot(gs[1, 2])
    models = [d['model'].upper() for d in all_data]
    times = [d['training_time']/60 for d in all_data]  # Convert to minutes
    bars = ax5.bar(models, times, color=colors[:len(all_data)], alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Training Time (min)', fontsize=11, fontweight='bold')
    ax5.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy improvement over epochs
    ax6 = fig.add_subplot(gs[2, 2])
    for idx, data in enumerate(all_data):
        final_acc = data['final_accuracy'] * 100
        best_acc = data['best_accuracy'] * 100
        ax6.barh(idx, final_acc, color=colors[idx], alpha=0.7, label='Final')
        ax6.plot([best_acc], [idx], marker='*', markersize=15, color='red')
    
    ax6.set_yticks(range(len(all_data)))
    ax6.set_yticklabels([d['model'].upper() for d in all_data])
    ax6.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Final Accuracy (★ = Best)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.legend(['Best Epoch', 'Final'], fontsize=9)
    
    fig.suptitle('ResNet Depth Comparison: ResNet10 vs ResNet18 vs ResNet50', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_name}.png")
    
    plt.show()

def print_detailed_summary(result_files):
    """Print detailed comparison summary"""
    print("\n" + "="*90)
    print("RESNET DEPTH COMPARISON - DETAILED SUMMARY")
    print("="*90)
    
    all_data = []
    for f in result_files:
        if os.path.exists(f):
            all_data.append(load_results(f))
    
    if not all_data:
        return
    
    # Header
    print(f"{'Model':<12} {'Params':<12} {'Best Acc':<12} {'Final Acc':<12} {'Final Loss':<12} {'Time(min)':<12}")
    print("-"*90)
    
    for data in all_data:
        model = data['model'].upper()
        params = f"{data['config']['total_params']/1e6:.2f}M"
        best_acc = f"{data['best_accuracy']*100:.2f}%"
        final_acc = f"{data['final_accuracy']*100:.2f}%"
        final_loss = f"{data['final_loss']:.4f}"
        train_time = f"{data['training_time']/60:.2f}"
        
        print(f"{model:<12} {params:<12} {best_acc:<12} {final_acc:<12} {final_loss:<12} {train_time:<12}")
    
    print("="*90)
    
    # Analysis
    print("\nKEY OBSERVATIONS:")
    print("-"*90)
    
    # Best model by accuracy
    best_model = max(all_data, key=lambda x: x['best_accuracy'])
    print(f"✓ Best Accuracy: {best_model['model'].upper()} with {best_model['best_accuracy']*100:.2f}%")
    
    # Fastest training
    fastest = min(all_data, key=lambda x: x['training_time'])
    print(f"✓ Fastest Training: {fastest['model'].upper()} with {fastest['training_time']/60:.2f} min")
    
    # Parameter efficiency
    for data in all_data:
        acc_per_param = data['best_accuracy'] * 100 / (data['config']['total_params'] / 1e6)
        print(f"✓ {data['model'].upper()} Efficiency: {acc_per_param:.2f}% accuracy per million parameters")
    
    print("="*90 + "\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        result_files = sys.argv[1:]
    else:
        result_files = [
            'report/resnet10_result.json',
            'report/resnet18_result.json',
            'report/resnet50_result.json'
        ]
    
    print_detailed_summary(result_files)
    plot_depth_comparison(result_files)
