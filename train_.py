import torch
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from model import GCN
from focal_loss import FocalLoss
from utils import load_alpha_weights
from preprocessing.create_batch_dataset import PDB_Dataset
from tqdm import tqdm  # FIX the import
from utils import calculate_class_weights, save_alpha_weights, load_alpha_weights




# Load datasets from the pickle file
dataset_save_path = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/sachinthadata/datasets.pkl"

with open(dataset_save_path, 'rb') as f:
    datasets = pickle.load(f)

pdb_protBERT_dataset_train = datasets['train']
pdb_protBERT_dataset_test = datasets['test']
pdb_protBERT_dataset_valid = datasets['valid']

print(f"Loaded datasets: Train={len(pdb_protBERT_dataset_train)}, Test={len(pdb_protBERT_dataset_test)}, Valid={len(pdb_protBERT_dataset_valid)}")

# Constants
THRESHOLD = 0.5
BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 0.00001  

BEST_MODEL_PATH = f'/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/best_model.pth'
CLASS_WEIGHT_PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/alpha_weights_s_train_1.pkl"
PLOT_SAVE_PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/results"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Data Loaders
train_loader = DataLoader(pdb_protBERT_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(pdb_protBERT_dataset_test, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(pdb_protBERT_dataset_valid, batch_size=BATCH_SIZE, shuffle=False)

alpha = calculate_class_weights(pdb_protBERT_dataset_train, device)
save_alpha_weights(alpha, CLASS_WEIGHT_PATH)
alpha = load_alpha_weights(CLASS_WEIGHT_PATH)
print(f"Alpha weights: {alpha}")

# Replace inf with maximum finite value
max_finite = alpha[alpha != float('inf')].max().item()
alpha[alpha == float('inf')] = max_finite
print(f"Cleaned Alpha weights: {alpha}")

# Model Setup
input_size = len(pdb_protBERT_dataset_train[0].x[0])
hidden_sizes = [1027, 912]
output_size = pdb_protBERT_dataset_train.num_classes
model = GCN(input_size, hidden_sizes, output_size).to(device)

# Criterion and Optimizer
criterion = FocalLoss(alpha=alpha, gamma=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# Tracking Metrics
train_losses, test_losses, valid_losses = [], [], []
train_accs, test_accs, valid_accs = [], [], []
best_valid_acc = 0

# Train Function
def train():
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in tqdm(train_loader, total=len(pdb_protBERT_dataset_train)):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = torch.sigmoid(out) > THRESHOLD
        correct += (pred == data.y).sum().item()
        total += np.prod(data.y.shape)

    '''        if i % 10 == 0:
                print(f"Batch {i}: Loss={loss.item():.4f}")
    '''
    return total_loss / len(train_loader), correct / total

# Test/Validation Function
def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.float())
            total_loss += loss.item()
            pred = torch.sigmoid(out) > THRESHOLD
            correct += (pred == data.y).sum().item()
            total += np.prod(data.y.shape)
    return total_loss / len(loader), correct / total

# Training Loop
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train()
    test_loss, test_acc = evaluate(test_loader)
    valid_loss, valid_acc = evaluate(valid_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    valid_losses.append(valid_loss)

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    valid_accs.append(valid_acc)

    # Update Scheduler
    scheduler.step(valid_loss)

    # Save Best Model
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved with Valid Acc: {valid_acc:.4f}")

    print(f"Epoch {epoch:03d}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Valid Acc: {valid_acc:.4f}")
    print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Valid Loss: {valid_loss:.4f}")

# **Plot Losses**
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='red')
plt.plot(valid_losses, label='Valid Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.grid(True)
plt.savefig(f"{PLOT_SAVE_PATH}/loss_plot.png")

# **Plot Accuracies**
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc', color='blue')
plt.plot(test_accs, label='Test Acc', color='red')
plt.plot(valid_accs, label='Valid Acc', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.grid(True)
plt.savefig(f"{PLOT_SAVE_PATH}/accuracy_plot.png")

# Show the plots
plt.show()
