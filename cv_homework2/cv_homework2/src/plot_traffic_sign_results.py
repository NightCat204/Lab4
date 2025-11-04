import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_training_results(results_file, output_file):
    """Visualize traffic sign classification training results"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    config = data['config']
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training and Test Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.7)
    ax1.plot(epochs, history['test_loss'], 'r-', linewidth=2, label='Test Loss', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and Test Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_accuracy'], 'b-', linewidth=2, label='Train Accuracy', alpha=0.7)
    ax2.plot(epochs, history['test_accuracy'], 'r-', linewidth=2, label='Test Accuracy', alpha=0.7)
    ax2.axhline(y=data['best_accuracy'], color='g', linestyle='--', 
                label=f'Best: {data["best_accuracy"]:.2f}%', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Schedule
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule (CosineAnnealing)', fontsize=13, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Overfitting Analysis
    ax4 = fig.add_subplot(gs[1, 0])
    gap = [train - test for train, test in zip(history['train_accuracy'], history['test_accuracy'])]
    colors = ['green' if g <= 5 else 'orange' if g <= 10 else 'red' for g in gap]
    ax4.bar(epochs, gap, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Warning (5%)')
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Overfitting (10%)')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Train-Test Gap (%)', fontsize=12)
    ax4.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Accuracy Improvement Rate
    ax5 = fig.add_subplot(gs[1, 1])
    test_acc_diff = [0] + [history['test_accuracy'][i] - history['test_accuracy'][i-1] 
                           for i in range(1, len(history['test_accuracy']))]
    smoothed = np.convolve(test_acc_diff, np.ones(5)/5, mode='same')
    ax5.plot(epochs, test_acc_diff, 'b-', alpha=0.3, label='Raw')
    ax5.plot(epochs, smoothed, 'r-', linewidth=2, label='Smoothed (5-epoch)')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Accuracy Change (%)', fontsize=12)
    ax5.set_title('Test Accuracy Improvement Rate', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Statistics Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['Dataset', data['dataset']],
        ['Classes', str(data['num_classes'])],
        ['Train Samples', str(data['train_samples'])],
        ['Test Samples', str(data['test_samples'])],
        ['', ''],
        ['Best Accuracy', f"{data['best_accuracy']:.2f}%"],
        ['Final Accuracy', f"{data['final_accuracy']:.2f}%"],
        ['Training Time', f"{data['training_time']/60:.1f} min"],
        ['', ''],
        ['Model', config.get('model', 'ResNet18')],
        ['Batch Size', str(config['batch_size'])],
        ['Learning Rate', str(config['lr'])],
        ['Optimizer', config['optimizer']],
        ['Scheduler', config.get('scheduler', 'N/A')],
        ['Label Smoothing', str(config.get('label_smoothing', 'N/A'))],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)  # Reduce height to avoid overlap
    
    # Style header
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(2):
            cell = table[(i, j)]
            if table_data[i][0] == '':
                cell.set_facecolor('#FFFFFF')
            elif i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
    
    ax6.set_title('Training Summary', fontsize=12, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle(f'Traffic Sign Classification - ResNet18\n'
                 f'Best Accuracy: {data["best_accuracy"]:.2f}% | Classes: {data["num_classes"]}',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Print class statistics
    print(f"\n{'='*70}")
    print(f"TRAFFIC SIGN CLASSIFICATION RESULTS")
    print(f"{'='*70}")
    print(f"Classes: {', '.join(data['class_names'])}")
    print(f"\nBest Test Accuracy: {data['best_accuracy']:.2f}%")
    print(f"Final Test Accuracy: {data['final_accuracy']:.2f}%")
    print(f"Training Time: {data['training_time']/60:.2f} minutes")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Traffic Sign Training Results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to training results JSON')
    parser.add_argument('--output', type=str, default='report/traffic_sign_training.png',
                        help='Output visualization path')
    args = parser.parse_args()
    
    plot_training_results(args.results, args.output)
