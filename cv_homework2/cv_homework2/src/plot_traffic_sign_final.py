import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_results(results_path, output_path):
    """Visualize training results in 2x2 layout"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    history = results['history']
    config = results['config']
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Training and Test Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=1.5, label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', linewidth=1.5, label='Test Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Test Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Training and Test Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_accuracy'], 'b-', linewidth=1.5, label='Train Accuracy')
    ax2.plot(epochs, history['test_accuracy'], 'r-', linewidth=1.5, label='Test Accuracy')
    ax2.axhline(y=results['best_accuracy'], color='g', linestyle='--', 
                linewidth=1.5, label=f'Best: {results["best_accuracy"]:.2f}%')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training and Test Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning Rate Schedule
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, history['learning_rate'], 'purple', linewidth=1.5)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Learning Rate', fontsize=11)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Test Accuracy Improvement Rate
    ax4 = fig.add_subplot(gs[1, 1])
    test_acc_diff = np.diff(history['test_accuracy'])
    smoothed_diff = np.convolve(test_acc_diff, np.ones(5)/5, mode='valid')
    ax4.plot(range(2, len(test_acc_diff)+1), test_acc_diff[1:], 
             'gray', alpha=0.4, linewidth=1, label='Raw')
    ax4.plot(range(3, len(smoothed_diff)+3), smoothed_diff, 
             'purple', linewidth=2, label='Smoothed (window=5)')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Accuracy Change (%)', fontsize=11)
    ax4.set_title('Test Accuracy Improvement Rate', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Main title
    title = f'Traffic Sign Classification Training Results\n'
    title += f'Model: {config["model"].upper()} | '
    title += f'Best Acc: {results["best_accuracy"]:.2f}% | '
    title += f'Final Acc: {results["final_accuracy"]:.2f}% | '
    title += f'Time: {results["training_time"]/60:.2f} min'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {output_path}')
    
    print(f'\nResults Summary:')
    print(f'  Model: {config["model"].upper()}')
    print(f'  Best Test Accuracy: {results["best_accuracy"]:.2f}%')
    print(f'  Final Test Accuracy: {results["final_accuracy"]:.2f}%')
    print(f'  Training Time: {results["training_time"]/60:.2f} minutes')
    print(f'  Total Epochs: {config["epochs"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    plot_results(args.results, args.output)
