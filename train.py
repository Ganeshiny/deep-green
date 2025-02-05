import torch
#from preprocessing.create_batch_dataset import PDB_Dataset
from torch_geometric.loader import DataLoader
from model import GCN, RareLabelGNN, GCN2
from sklearn.model_selection import train_test_split
import numpy as np
from focal_loss import FocalLoss
from utils import calculate_class_weights, save_alpha_weights, load_alpha_weights
import pickle
import json

# Load datasets from the pickle file
dataset_save_path = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/pdb_datasets.pkl"

with open(dataset_save_path, 'rb') as f:
    datasets = pickle.load(f)

pdb_protBERT_dataset_train = datasets['train']
pdb_protBERT_dataset_test = datasets['test']
pdb_protBERT_dataset_valid = datasets['valid']

print(f"Loaded datasets: Train={len(pdb_protBERT_dataset_train)}, Test={len(pdb_protBERT_dataset_test)}, Valid={len(pdb_protBERT_dataset_valid)}")


# Constants
THRESHOLD = 0.5
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.00001 
BEST_MODEL_PATH = f'/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_weights_GCN2_{EPOCHS}_epochs_{BATCH_SIZE}_2_layers_cross.pth'
PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_GCN2.pth"
CLASS_WEIGHT_PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/alpha_weights.pkl"
MODEL_INFO_PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_info_GCN2_2_layers.json"  # Path to save model info

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Dataset Setup
root = 'preprocessing/data/structure_files/tmp_cmap_files'
annot_file = 'preprocessing/data/pdb2go.tsv'
num_shards = 20

# Load datasets from the pickle file
dataset_save_path = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/pdb_datasets.pkl"

with open(dataset_save_path, 'rb') as f:
    datasets = pickle.load(f)

pdb_protBERT_dataset_train = datasets['train']
pdb_protBERT_dataset_test = datasets['test']
pdb_protBERT_dataset_valid = datasets['valid']

print(f"Loaded datasets: Train={len(pdb_protBERT_dataset_train)}, Test={len(pdb_protBERT_dataset_test)}, Valid={len(pdb_protBERT_dataset_valid)}")

print(f'Number of training graphs: {len(pdb_protBERT_dataset_train)}')
print(f'Number of test graphs: {len(pdb_protBERT_dataset_test)}')

train_loader = DataLoader(pdb_protBERT_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(pdb_protBERT_dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# Calculate alpha
#alpha = calculate_class_weights(dataset, device)
#save_alpha_weights(alpha, CLASS_WEIGHT_PATH)
alpha = load_alpha_weights(CLASS_WEIGHT_PATH)
print(f"Alpha weights:{alpha}")

# Model Setup
input_size = len(pdb_protBERT_dataset_train[0].x[0])
print(f"input size: {input_size}")
hidden_sizes = [1027, 912]
output_size = pdb_protBERT_dataset_train.num_classes
print(f"output size: {output_size}")
model = GCN2(input_size, hidden_sizes, output_size)
model.to(device)

model_info = {
    "input_size": input_size,
    "hidden_sizes": hidden_sizes,
    "output_size": output_size
}

with open(MODEL_INFO_PATH, 'w') as f:
    json.dump(model_info, f)

torch.save(model.state_dict(), PATH)

# Criterion and Optimizer
# Use focal loss with class weights
criterion = FocalLoss(alpha=alpha, gamma=3)

# Optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# Accumulate gradients over multiple smaller batches
accumulation_steps = 4  # Accumulate over 4 smaller batches

def train():
    model.train()
    optimizer.zero_grad()  # Reset gradients
    for i, data in enumerate(train_loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.float())
        loss = loss / accumulation_steps  # Scale loss
        loss.backward()

        # Perform optimizer step every accumulation_steps batches
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


def test(loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = torch.sigmoid(out) > THRESHOLD  # Convert to probabilities and threshold
            correct += (pred == data.y).sum().item()  # Count correct predictions
            total += np.prod(data.y.shape)  # Total number of labels
    return correct / total  # Accuracy across all labels

# Tracking best accuracy
best_test_acc = 0

for epoch in range(1, EPOCHS + 1):
    # Train the model for one epoch
    train()
    
    # Evaluate the model
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    
    # Scheduler step based on test accuracy (or loss)
    scheduler.step(1 - test_acc)  # For accuracy, pass `1 - test_acc` (higher is better)
    # If monitoring test loss instead, pass the actual test loss

    # Save the model if test accuracy improves
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved with Test Acc: {test_acc:.4f}")
    
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

