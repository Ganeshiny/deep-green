import torch
from preprocessing.create_batch_dataset import PDB_Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model import GCN
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from focal_loss import FocalLoss

# Constants
THRESHOLD = 0.5
BATCH_SIZE = 64
FOLDS = 5
EPOCHS = 5

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Dataset Setup
root = 'preprocessing/data/structure_files/tmp_cmap_files'
annot_file = 'preprocessing/data/pdb2go.tsv'
num_shards = 20

torch.manual_seed(12345)
pdb_protBERT_dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process", model="protBERT")
pdb_baseline_dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process")

print()
print(f'Dataset: {pdb_protBERT_dataset}:')
print('====================')
print(f'Number of graphs: {len(pdb_protBERT_dataset)}')
print(f'Number of features: {pdb_protBERT_dataset.num_features}')
print(f'Number of classes: {pdb_protBERT_dataset.num_classes}')

data = pdb_protBERT_dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


# Model Setup
input_size = len(pdb_protBERT_dataset[0].x[0])
print(f"input size {input_size}")
hidden_sizes = [1000, 912, 820, 500]
output_size = pdb_protBERT_dataset.num_classes
print(f"output size {output_size}")
model = GCN(input_size, hidden_sizes, output_size)
print(f"model: {model}")
model.to(device)
torch.save(model.state_dict(), 'model_GCN_protBERT.pth')

# Criterion and Optimizer
criterion = FocalLoss()
optimizer = optim.Adadelta(model.parameters())

def train(fold, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        target = data.y.float().to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Correct progress tracking
        if batch_idx % 10 == 0:  # Adjust frequency of logging
            current_samples = batch_idx * len(data.y)
            total_samples = len(train_loader.dataset)
            progress = 100. * current_samples / total_samples
            print(f'Train Fold/Epoch: {fold}/{epoch} [{current_samples}/{total_samples} ({progress:.0f}%)]\tLoss: {loss.item():.6f}')

def test(fold, model, device, test_loader):
    model.eval()
    test_loss = 0
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            test_loss += criterion(output, data.y.float()).item()
            pred = torch.sigmoid(output)
            all_val_preds.append(pred.cpu().numpy())
            all_val_labels.append(data.y.cpu().numpy())

    test_loss /= len(test_loader)
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)

    accuracy = ((all_val_preds > 0.5) == all_val_labels).mean()
    print(f'\nTest set for fold {fold}: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}\n')

# K-Fold Cross-Validation
kfold = KFold(n_splits=FOLDS, shuffle=True)
'''
for fold, (train_idx, test_idx) in enumerate(kfold.split(pdb_protBERT_dataset)):
    print(f'------------ Fold {fold} ------------')
    
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    
    trainloader = DataLoader(pdb_protBERT_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
    testloader = DataLoader(pdb_protBERT_dataset, batch_size=BATCH_SIZE, sampler=test_subsampler)

    for epoch in range(1, EPOCHS + 1):
        train(fold, model, device, trainloader, optimizer, epoch)
        test(fold, model, device, testloader)
'''