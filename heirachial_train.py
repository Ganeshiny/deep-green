import torch
from preprocessing.create_batch_dataset import PDB_Dataset
from torch_geometric.loader import DataLoader
from model_heirachial import HierarchicalGNN
from sklearn.model_selection import train_test_split
import numpy as np
from utils import calculate_class_weights, save_alpha_weights, load_alpha_weights
import pickle
import json
from focal_loss import HierarchicalFocalLoss

# Constants
THRESHOLD = 0.5
BATCH_SIZE = 256
EPOCHS = 500
LEARNING_RATE = 0.00001
ACCUMULATION_STEPS = 4  # Accumulate gradients over smaller batches

# Paths
BASE_PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor"
BEST_MODEL_PATH = f"{BASE_PATH}/model_and_weight_files/best_model.pth"
MODEL_INFO_PATH = f"{BASE_PATH}/model_and_weight_files/model_info.json"
CLASS_WEIGHT_PATH = f"{BASE_PATH}/model_and_weight_files/alpha_weights.pkl"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Dataset Setup
root = f"{BASE_PATH}/preprocessing/data/structure_files/tmp_cmap_files"
annot_file = f"{BASE_PATH}/preprocessing/data/pdb2go.tsv"
num_shards = 20

torch.manual_seed(12345)
pdb_protBERT_dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process", model="protBERT")
dataset = pdb_protBERT_dataset.shuffle()

# Split dataset
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=12345)
print(f'Training graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Compute Class Weights for Focal Loss
alpha = calculate_class_weights(dataset, device)
save_alpha_weights(alpha, CLASS_WEIGHT_PATH)
alpha = load_alpha_weights(CLASS_WEIGHT_PATH)
print(f"Alpha weights: {alpha}")

# Model Setup
input_size = len(pdb_protBERT_dataset[0].x[0])
hidden_sizes = [1024, 512]
output_size = pdb_protBERT_dataset.num_classes

# Load GO Hierarchy (for Hierarchical Loss)
hierarchy_matrix = torch.load(f"{BASE_PATH}/preprocessing/data/go_hierarchy_matrix.pt").to(device)

# Initialize Model
model = HierarchicalGNN(input_size, hidden_sizes, output_size, go_hierarchy=hierarchy_matrix)
model.to(device)

# Save model metadata
model_info = {"input_size": input_size, "hidden_sizes": hidden_sizes, "output_size": output_size}
with open(MODEL_INFO_PATH, 'w') as f:
    json.dump(model_info, f)

# Loss and Optimizer
criterion = HierarchicalFocalLoss(alpha=alpha, hierarchy_matrix=hierarchy_matrix)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# **Training Function with Gradient Accumulation**
def train():
    model.train()
    optimizer.zero_grad()  # Reset gradients
    
    for i, data in enumerate(train_loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.float()) / ACCUMULATION_STEPS  # Scale loss

        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

# **Testing Function**
def test(loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = torch.sigmoid(out) > THRESHOLD
            correct += (pred == data.y).sum().item()
            total += np.prod(data.y.shape)
    return correct / total  # Accuracy

# **Training Loop**
best_test_acc = 0
for epoch in range(1, EPOCHS + 1):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    # Reduce learning rate if test accuracy stagnates
    scheduler.step(1 - test_acc)

    # Save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved with Test Acc: {test_acc:.4f}")
    
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
