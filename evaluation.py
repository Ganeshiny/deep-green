import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from torch_geometric.loader import DataLoader
import json
import argparse
from model import HybridGCNGAT
from preprocessing.create_batch_dataset import PDB_Dataset
import pickle
import pandas as pd
from utils import compute_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the current script directory and create output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "PFP_evaluation_metrics")
os.makedirs(output_dir, exist_ok=True)

# Load datasets from the pickle file
dataset_save_path = "./protein-go-predictor/preprocessing/data/sachinthadata/MFO_datasets.pkl"

with open(dataset_save_path, 'rb') as f:
    datasets = pickle.load(f)

pdb_protBERT_dataset_train = datasets['train']
pdb_protBERT_dataset_test = datasets['test']
pdb_protBERT_dataset_valid = datasets['valid']

# Model Setup
input_size = len(pdb_protBERT_dataset_train[0].x[0])
hidden_sizes =[1024, 512]
output_size = pdb_protBERT_dataset_train.num_classes


print(f"Loaded datasets: Train={len(pdb_protBERT_dataset_train)}, Test={len(pdb_protBERT_dataset_test)}, Valid={len(pdb_protBERT_dataset_valid)}")

test_loader = DataLoader(pdb_protBERT_dataset_test, shuffle=False)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = torch.sigmoid(out)
            all_preds.append(pred.cpu())
            all_targets.append(data.y.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    metrics, thresholds = compute_metrics(all_targets, all_preds)
    return metrics, thresholds, all_targets, all_preds

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
    """Save enhanced violin plot with swarm overlay"""
    valid_aupr = ~np.isnan(metrics['aupr_per_label'])
    valid_fmax = ~np.isnan(metrics['fmax_per_label'])
    
    aupr_data = metrics['aupr_per_label'][valid_aupr]
    fmax_data = metrics['fmax_per_label'][valid_fmax]
    
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame for seaborn
    data = pd.DataFrame({
        'Metric': ['AUPR']*len(aupr_data) + ['Fmax']*len(fmax_data),
        'Score': np.concatenate([aupr_data, fmax_data])
    })
    
    # Create violin plot
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
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot to: {output_path}")

def save_boxplot(metrics, filename):
    """Save enhanced boxplot visualization"""
    valid_aupr = ~np.isnan(metrics['aupr_per_label'])
    valid_fmax = ~np.isnan(metrics['fmax_per_label'])
    
    aupr_data = metrics['aupr_per_label'][valid_aupr]
    fmax_data = metrics['fmax_per_label'][valid_fmax]
    
    plt.figure(figsize=(10, 6))
    
    # Create DataFrame for seaborn
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
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot to: {output_path}")

def load_model(model_path, model_info_path, device):
    """Load model from checkpoint"""
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    model = HybridGCNGAT(
        input_size = input_size,         
        hidden_sizes= hidden_sizes,
        output_size=output_size,         
        gcn_layers=1,            
        gat_layers=1,            
        dropout_rate=0.3
    ).to(device)


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./protein-go-predictor/model_and_weight_files/WEIGHTS_2LAYERS_mfo.pth')
    parser.add_argument('--annot_dict', type=str, default='./protein-go-predictor/preprocessing/data/annot_dict.pkl')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()

    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(pdb_protBERT_dataset_train[0].x[0])
    hidden_sizes =[1024, 512]
    output_size = pdb_protBERT_dataset_train.num_classes

    # Initialize and load model
    model = HybridGCNGAT(
    input_size = input_size,         
    hidden_sizes= hidden_sizes,
    output_size=output_size,         
    gcn_layers=1,            
    gat_layers=1,            
    dropout_rate=0.3
    ).to(device)    

    model.load_state_dict(torch.load(args.model_path))
    model.to(device).eval()

    # Evaluate Model
    metrics, thresholds, y_true, y_pred = evaluate_model(model, test_loader, device)

    # Print Results
    print("\n**Evaluation Results:**")
    print(f"Micro AUPR: {metrics['aupr_micro']:.4f}")
    print(f"Macro AUPR: {metrics['aupr_macro']:.4f}")
    #print(f"AUPR per label: {metrics['aupr_per_label']:.4f}")
    print(f"Micro Fmax: {metrics['fmax_micro']:.4f} (Best T: {metrics['best_t_micro']:.2f})")
    print(f"Macro Fmax: {metrics['fmax_macro']:.4f} (Best T: {metrics['best_t_macro']:.2f})")
    # Find tuple for threshold 0.5 (or nearest if not exact)
    at_05 = min(metrics['fmax_values'], key=lambda x: abs(x[0] - 0.5))
    micro_at_05 = at_05[1]
    macro_at_05 = at_05[2]

    print(f"Micro Fmax: {metrics['fmax_micro']:.4f} (at 0.5: {micro_at_05:.4f})")
    print(f"Macro Fmax: {metrics['fmax_macro']:.4f} (at 0.5: {macro_at_05:.4f})")

    # Save all visualizations
    save_combined_plot(y_true, y_pred, thresholds, metrics, "combined_metrics.png")
    save_violin_plot(metrics, "enhanced_violin_plot.png")
    save_boxplot(metrics, "enhanced_boxplot.png")
    
    print(f"\nAll evaluation plots saved to: {output_dir}")

if __name__ == '__main__':
    main()