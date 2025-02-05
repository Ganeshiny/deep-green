# evaluate.py
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch_geometric.loader import DataLoader
import json
import pickle
import argparse
from model import GCN
from preprocessing.create_batch_dataset import PDB_Dataset
import torch
from preprocessing.create_batch_dataset import PDB_Dataset
from torch_geometric.loader import DataLoader
from model import GCN, RareLabelGNN, GCN2
from sklearn.model_selection import train_test_split
import numpy as np
from focal_loss import FocalLoss
from utils import calculate_class_weights, save_alpha_weights, load_alpha_weights
import pickle
import json

'''
def calculate_metrics(y_true, y_pred):
    """Calculate micro/macro AUPR and Fmax"""
    metrics = {}
    
    # AUPR calculations
    metrics['micro_aupr'] = average_precision_score(y_true, y_pred, average='micro')
    metrics['macro_aupr'] = average_precision_score(y_true, y_pred, average='macro')
    
    # Fmax calculation (CAFA-style)
    thresholds = np.arange(0.0, 1.0, 0.01)
    fmax = 0
    best_t = 0
    
    for t in thresholds:
        y_pred_t = (y_pred > t).astype(int)
        tp = np.sum(y_true * y_pred_t)
        fp = np.sum(y_pred_t) - tp
        fn = np.sum(y_true) - tp
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if f1 > fmax:
            fmax = f1
            best_t = t
            
    metrics['fmax'] = fmax
    metrics['best_threshold'] = best_t
    
    return metrics

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = torch.sigmoid(out)
            
            all_preds.append(pred.cpu())
            all_targets.append(data.y.cpu())
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    return calculate_metrics(all_targets, all_preds)

def load_model(model_path, model_info_path, device):
    # Load model info
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Initialize model
    model = GCN(
        input_size=model_info['input_size'],
        hidden_sizes=model_info['hidden_sizes'],
        output_size=model_info['output_size']
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_weights_200_epochs_128_2_layers_cross.pth', help = "")
    parser.add_argument('--model_info', type=str, default='/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_info_2_layers.json' , help = "")
    parser.add_argument('--annot_dict', type=str, default='protein-go-predictor/preprocessing/data/annot_dict.pkl' , help = "")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(args.model_path, args.model_info, device)
    
    # Dataset Setup
    root = '/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/structure_files/tmp_cmap_files'
    annot_file = '/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/pdb2go.tsv'
    num_shards = 20

    torch.manual_seed(12345)
    pdb_protBERT_dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process", model="protBERT")
    dataset = pdb_protBERT_dataset.shuffle()

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=12345)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Micro AUPR: {metrics['micro_aupr']:.4f}")
    print(f"Macro AUPR: {metrics['macro_aupr']:.4f}")
    print(f"Fmax: {metrics['fmax']:.4f} (Best threshold: {metrics['best_threshold']:.2f})")

if __name__ == '__main__':
    main()'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from torch_geometric.loader import DataLoader
import json
import argparse
from model import GCN
from preprocessing.create_batch_dataset import PDB_Dataset
from sklearn.model_selection import train_test_split


def calculate_metrics(y_true, y_pred):
    """Calculate micro/macro AUPR and Fmax"""
    metrics = {}

    # **AUPR Calculations**
    metrics['micro_aupr'] = average_precision_score(y_true, y_pred, average='micro')
    metrics['macro_aupr'] = average_precision_score(y_true, y_pred, average='macro')

    # **Micro & Macro Fmax Calculation (CAFA-style)**
    thresholds = np.arange(0.0, 1.0, 0.01)
    
    fmax_micro, best_t_micro = 0, 0
    fmax_macro, best_t_macro = 0, 0

    for t in thresholds:
        y_pred_t = (y_pred > t).astype(int)

        # **Micro Fmax**
        f1_micro = f1_score(y_true, y_pred_t, average='micro')
        if f1_micro > fmax_micro:
            fmax_micro, best_t_micro = f1_micro, t

        # **Macro Fmax**
        f1_macro = f1_score(y_true, y_pred_t, average='macro')
        if f1_macro > fmax_macro:
            fmax_macro, best_t_macro = f1_macro, t

    metrics['micro_fmax'], metrics['best_t_micro'] = fmax_micro, best_t_micro
    metrics['macro_fmax'], metrics['best_t_macro'] = fmax_macro, best_t_macro

    return metrics, thresholds


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

    # **Convert to Numpy**
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # **Compute Metrics**
    metrics, thresholds = calculate_metrics(all_targets, all_preds)

    return metrics, thresholds, all_targets, all_preds


def plot_results(y_true, y_pred, thresholds, metrics, output_file="evaluation_results.png"):
    """Visualize Precision-Recall Curve and Fmax vs Threshold"""

    # **Precision-Recall Curve**
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f"Micro AUPR = {metrics['micro_aupr']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    # **Fmax vs Threshold Curve**
    plt.subplot(1, 2, 2)
    fmax_values = [f1_score(y_true, (y_pred > t).astype(int), average="micro") for t in thresholds]
    plt.plot(thresholds, fmax_values, label="Micro Fmax", color="blue")

    fmax_values_macro = [f1_score(y_true, (y_pred > t).astype(int), average="macro") for t in thresholds]
    plt.plot(thresholds, fmax_values_macro, label="Macro Fmax", color="red")

    plt.axvline(metrics['best_t_micro'], color="blue", linestyle="--", label=f"Best Micro T: {metrics['best_t_micro']:.2f}")
    plt.axvline(metrics['best_t_macro'], color="red", linestyle="--", label=f"Best Macro T: {metrics['best_t_macro']:.2f}")

    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Fmax vs Threshold")
    plt.legend()

    # **Save & Show**
    plt.savefig(output_file)
    plt.show()


def load_model(model_path, model_info_path, device):
    """Load model from checkpoint"""
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    model = GCN(
        input_size=model_info['input_size'],
        hidden_sizes=model_info['hidden_sizes'],
        output_size=model_info['output_size']
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_weights_200_epochs_128_2_layers_cross.pth')
    parser.add_argument('--model_info', type=str, default='/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_info_2_layers.json')
    parser.add_argument('--annot_dict', type=str, default='/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/annot_dict.pkl')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_plot', type=str, default='evaluation_results.png', help="Path to save visualization")
    args = parser.parse_args()

    # **Device Setup**
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # **Load Model**
    model = load_model(args.model_path, args.model_info, device)

    # **Dataset Setup**
    root = '/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/structure_files/tmp_cmap_files'
    annot_file = '/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/pdb2go.tsv'
    num_shards = 20

    torch.manual_seed(12345)
    pdb_protBERT_dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process", model="protBERT")
    dataset = pdb_protBERT_dataset.shuffle()

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=12345)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # **Evaluate Model**
    metrics, thresholds, y_true, y_pred = evaluate_model(model, test_loader, device)

    # **Print Results**
    print("\n**Evaluation Results:**")
    print(f" **Micro AUPR:** {metrics['micro_aupr']:.4f}")
    print(f" **Macro AUPR:** {metrics['macro_aupr']:.4f}")
    print(f" **Micro Fmax:** {metrics['micro_fmax']:.4f} (Best T: {metrics['best_t_micro']:.2f})")
    print(f" **Macro Fmax:** {metrics['macro_fmax']:.4f} (Best T: {metrics['best_t_macro']:.2f})")

    # **Visualize Results**
    plot_results(y_true, y_pred, thresholds, metrics, args.output_plot)


if __name__ == '__main__':
    main()
