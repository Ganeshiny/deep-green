import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import compute_metrics
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

output_dir = "deepfri_evals/deepfri_evaluation_plots"
os.makedirs(output_dir, exist_ok=True)

# Load the pickle file that has the predicted and actual label of test dataset
with open('/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/deepfri_evals/GCN-PDB_BP_results.pckl', 'rb') as f:
    data = pickle.load(f)
    y_true, y_pred = data['Y_true'], data['Y_pred']
y_true = np.array(y_true)
y_pred = np.array(y_pred)

def save_combined_plot(y_true, y_pred, thresholds, metrics, filename):
    """Save enhanced combined evaluation plot"""
    plt.figure(figsize=(18, 6))
    
    # 1. Precision-Recall Curve
    plt.subplot(1, 2, 1)
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    plt.plot(recall, precision, label=f'Micro AUPR: {metrics["aupr_micro"]:.3f}', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Average Precision-Recall Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    '''    # 2. Fmax vs Threshold
        plt.subplot(1, 3, 2)
        fmax_values = [f1_score(y_true, (y_pred > t).astype(int), average="micro") for t in thresholds]
        plt.plot(thresholds, fmax_values, label="Micro Fmax", color="blue", linewidth=2)
        fmax_values_macro = [f1_score(y_true, (y_pred > t).astype(int), average="macro") for t in thresholds]
        plt.plot(thresholds, fmax_values_macro, label="Macro Fmax", color="red", linewidth=2)
        plt.axvline(metrics['best_t_micro'], color="blue", linestyle="--", label=f'Best Micro: {metrics["best_t_micro"]:.2f}')
        plt.axvline(metrics['best_t_macro'], color="red", linestyle="--", label=f'Best Macro: {metrics["best_t_macro"]:.2f}')
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.title("Fmax vs Threshold", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)'''
    
    # 3. Violin plots
    plt.subplot(1, 2, 2)
    valid_aupr = ~np.isnan(metrics['aupr_per_label'])
    valid_fmax = ~np.isnan(metrics['fmax_per_label'])
    
    violin_parts = plt.violinplot(
        [metrics['aupr_per_label'][valid_aupr], 
        metrics['fmax_per_label'][valid_fmax]], 
        positions=[1, 2],
        showmeans=True,
        showmedians=True
    )
    
    colors = ['#1f77b4', '#ff7f0e']
    for pc, color in zip(violin_parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    plt.xticks([1, 2], ['AUPR', 'Fmax'], fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Per-Label Performance Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to: {output_path}")

    
def save_violin_plot(metrics, filename):
    """Save enhanced violin plot"""
    valid_aupr = ~np.isnan(metrics['aupr_per_label'])
    valid_fmax = ~np.isnan(metrics['fmax_per_label'])
    
    aupr_data = metrics['aupr_per_label'][valid_aupr]
    fmax_data = metrics['fmax_per_label'][valid_fmax]
    
    plt.figure(figsize=(12, 6))
    
    # Create a DataFrame for seaborn
    import pandas as pd
    data = pd.DataFrame({
        'Metric': ['AUPR']*len(aupr_data) + ['Fmax']*len(fmax_data),
        'Score': np.concatenate([aupr_data, fmax_data])
    })
    
    # Create violin plot with seaborn
    sns.violinplot(x='Metric', y='Score', data=data, palette=['#1f77b4', '#ff7f0e'],
                  inner='quartile', cut=0)
    
    # Add swarm plot
    sns.swarmplot(x='Metric', y='Score', data=data, color='black', alpha=0.3)
    
    plt.title('Distribution of Performance Metrics per Label', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def save_boxplot(metrics, filename):
    """Save enhanced boxplot"""
    valid_aupr = ~np.isnan(metrics['aupr_per_label'])
    valid_fmax = ~np.isnan(metrics['fmax_per_label'])
    
    aupr_data = metrics['aupr_per_label'][valid_aupr]
    fmax_data = metrics['fmax_per_label'][valid_fmax]
    
    plt.figure(figsize=(10, 6))
    
    # Create a DataFrame for seaborn
    import pandas as pd
    data = pd.DataFrame({
        'Metric': ['AUPR']*len(aupr_data) + ['Fmax']*len(fmax_data),
        'Score': np.concatenate([aupr_data, fmax_data])
    })
    
    # Create boxplot
    sns.boxplot(x='Metric', y='Score', data=data, palette=['#1f77b4', '#ff7f0e'],
               showmeans=True, 
               meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"red"})
    
    plt.title('Boxplot of Performance Metrics per Label', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Compute metrics
metrics, _ = compute_metrics(y_true, y_pred)

# Print results
print(f'Best Fmax (Micro): {metrics["fmax_micro"]:.4f} at threshold {metrics["best_t_micro"]:.2f}')
print(f'Best Fmax (Macro): {metrics["fmax_macro"]:.4f} at threshold {metrics["best_t_macro"]:.2f}')
print(f'AUPR (Micro): {metrics["aupr_micro"]:.4f}')
print(f'AUPR (Macro): {metrics["aupr_macro"]:.4f}')

# Save all plots
save_combined_plot(metrics, y_true, y_pred, "combined_metrics.png")
save_violin_plot(metrics, "enhanced_violin_plot.png")
save_boxplot(metrics, "enhanced_boxplot.png")

print(f"\nAll plots saved to directory: {output_dir}")