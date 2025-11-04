import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_results(filename):
    """Load training results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_comparison(result_files, labels, output_name='report/lr_comparison'):
    """
    Plot comparison of training curves from multiple experiments
    
    Args:
        result_files: list of JSON result filenames
        labels: list of labels for each experiment
        output_name: prefix for output plot files
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Learning Rate Analysis: Training Dynamics Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(result_files)))
    
    for idx, (result_file, label) in enumerate(zip(result_files, labels)):
        if not os.path.exists(result_file):
            print(f"Warning: {result_file} not found, skipping...")
            continue
            
        data = load_results(result_file)
        epochs = range(1, len(data['epoch_losses']) + 1)
        color = colors[idx]
        
        # Plot 1: Training Loss vs Epoch
        axes[0, 0].plot(epochs, data['epoch_losses'], label=label, color=color, linewidth=2, marker='o', markersize=3)
        
        # Plot 2: Test Accuracy vs Epoch
        axes[0, 1].plot(epochs, data['epoch_accuracies'], label=label, color=color, linewidth=2, marker='s', markersize=3)
        
        # Plot 3: Learning Rate Schedule
        axes[1, 0].plot(epochs, data['learning_rates'], label=label, color=color, linewidth=2, marker='^', markersize=3)
        
        # Plot 4: Loss (log scale)
        axes[1, 1].semilogy(epochs, data['epoch_losses'], label=label, color=color, linewidth=2, marker='o', markersize=3)
    
    # Configure Plot 1: Training Loss
    axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Training Loss vs. Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Configure Plot 2: Test Accuracy
    axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Test Accuracy vs. Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Configure Plot 3: Learning Rate Schedule
    axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Configure Plot 4: Loss (log scale)
    axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Training Loss (log scale)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Training Loss (Log Scale)', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    
    plt.show()

def print_summary(result_files, labels):
    """Print summary statistics for all experiments"""
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<30} {'Initial LR':<12} {'Schedule':<10} {'Best Acc':<12} {'Final Loss':<12} {'Time(s)':<10}")
    print("-"*80)
    
    for result_file, label in zip(result_files, labels):
        if not os.path.exists(result_file):
            continue
            
        data = load_results(result_file)
        config = data['config']
        
        initial_lr = config['initial_lr']
        schedule = 'Yes' if config['lr_schedule'] else 'No'
        best_acc = data['best_accuracy']
        final_loss = data['epoch_losses'][-1]
        train_time = data['training_time']
        
        print(f"{label:<30} {initial_lr:<12.6f} {schedule:<10} {best_acc:<12.4f} {final_loss:<12.4f} {train_time:<10.2f}")
    
    print("="*80 + "\n")

if __name__ == '__main__':
    # Check if result files are provided as arguments
    if len(sys.argv) > 1:
        result_files = []
        labels = []
        
        # Parse arguments: --file1 label1 --file2 label2 ...
        i = 1
        while i < len(sys.argv):
            if sys.argv[i].startswith('--'):
                if i + 1 < len(sys.argv):
                    result_files.append(sys.argv[i+1])
                    labels.append(sys.argv[i].replace('--', '').replace('-', ' ').title())
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        if result_files:
            print_summary(result_files, labels)
            plot_comparison(result_files, labels)
        else:
            print("Usage: python plot_lr_comparison.py --exp1 result1.json --exp2 result2.json ...")
    else:
        # Default: look for predefined result files in report directory
        result_files = [
            'report/lr_0.001_fixed.json',
            'report/lr_0.01_fixed.json', 
            'report/lr_0.1_fixed.json',
            'report/lr_0.1_scheduled.json'
        ]
        
        labels = [
            'LR=0.001 (Fixed)',
            'LR=0.01 (Fixed)',
            'LR=0.1 (Fixed)',
            'LR=0.1 (Scheduled)'
        ]
        
        # Filter existing files
        existing_files = []
        existing_labels = []
        for f, l in zip(result_files, labels):
            if os.path.exists(f):
                existing_files.append(f)
                existing_labels.append(l)
        
        if existing_files:
            print_summary(existing_files, existing_labels)
            plot_comparison(existing_files, existing_labels)
        else:
            print("No result files found. Please run training experiments first.")
            print("\nExpected files:")
            for f in result_files:
                print(f"  - {f}")
