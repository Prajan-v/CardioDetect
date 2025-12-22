import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
REPORTS_DIR = Path('/Users/prajanv/CardioDetect/Milestone_2/reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_optuna_chart():
    # User Prompt Logic:
    # 50 random points, log scale X (0.01-0.1), Y (0.82-0.90)
    # Winning point: x=0.047, y=0.9160 (Gold, Larger)
    # Annotation: "Best: 91.60%\nLR: 0.047"
    # Red dashed line at y=0.9160
    
    np.random.seed(42)
    n_points = 50
    learning_rates = np.power(10, np.random.uniform(np.log10(0.01), np.log10(0.1), n_points))
    accuracies = np.random.uniform(0.82, 0.90, n_points)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Scatter plot of random points
    plt.scatter(learning_rates, accuracies, color='blue', alpha=0.6, label='Trials')
    
    # Winning Point
    best_lr = 0.047
    best_acc = 0.9160
    plt.scatter([best_lr], [best_acc], color='gold', s=150, edgecolors='black', label='Best Model', zorder=5)
    
    # Red dashed line
    plt.axhline(y=best_acc, color='red', linestyle='--', linewidth=1.5, label=f'Max Accuracy ({best_acc:.2%})')
    
    # Annotation
    plt.annotate(f'Best: {best_acc:.2%}\nLR: {best_lr}', 
                 xy=(best_lr, best_acc), 
                 xytext=(best_lr+0.01, best_acc-0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    plt.xscale('log')
    plt.title('Optuna Hyperparameter Tuning: Learning Rate vs. Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Learning Rate (log scale)', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    
    # Save
    output_path = REPORTS_DIR / 'optuna_param_vs_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Optuna Chart to: {output_path}")

if __name__ == "__main__":
    generate_optuna_chart()
