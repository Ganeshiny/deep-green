import torch
import numpy as np
import pickle
import json
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader
from model import GCN
from focal_loss import FocalLoss
from utils import load_alpha_weights
from preprocessing.create_batch_dataset import PDB_Dataset
from utils import calculate_class_weights, save_alpha_weights, load_alpha_weights

def calculate_metrics(y_true, y_pred):
    """Calculate CAFA-style metrics"""
    metrics = {
        'micro_aupr': average_precision_score(y_true, y_pred, average='micro'),
        'macro_aupr': average_precision_score(y_true, y_pred, average='macro'),
        'fmax': 0.0,
        'best_threshold': 0.0
    }
    
    # Fmax calculation
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_f1 = 0
    for t in thresholds:
        y_pred_t = (y_pred > t).astype(int)
        tp = np.sum(y_true * y_pred_t)
        fp = np.sum(y_pred_t) - tp
        fn = np.sum(y_true) - tp
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        if f1 > best_f1:
            best_f1 = f1
            metrics['fmax'] = f1
            metrics['best_threshold'] = t
            
    return metrics

def evaluate(loader, model, device):
    """Full evaluation with metric calculation"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = torch.sigmoid(out)
            
            all_preds.append(preds.cpu())
            all_targets.append(data.y.cpu())
    
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    return calculate_metrics(y_true, y_pred)

# Load datasets
dataset_save_path = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/sachintha data/pdb_datasets.pkl"
with open(dataset_save_path, 'rb') as f:
    datasets = pickle.load(f)

pdb_protBERT_dataset_train = datasets['train']
pdb_protBERT_dataset_test = datasets['test']
pdb_protBERT_dataset_valid = datasets['valid']

# Verify datasets
assert len(pdb_protBERT_dataset_train) > 0, "Train dataset is empty!"
assert len(pdb_protBERT_dataset_test) > 0, "Test dataset is empty!"
assert len(pdb_protBERT_dataset_valid) > 0, "Validation dataset is empty!"

# Training parameters
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-5
ACCUMULATION_STEPS = 4
PATIENCE = 10  # For early stopping

BEST_MODEL_PATH = f'/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_weights_GCN2_{EPOCHS}_epochs_{BATCH_SIZE}_2_layers_cross.pth'
PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_GCN2.pth"
CLASS_WEIGHT_PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/alpha_weights_s_train_1.pkl"
MODEL_INFO_PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_info_GCN2_2_layers.json"  # Path to save model info

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize model
input_size = len(pdb_protBERT_dataset_train[0].x[0])
hidden_sizes = [1027, 912]
output_size = pdb_protBERT_dataset_train.num_classes

model = GCN(input_size, hidden_sizes, output_size).to(device)

alpha_train = calculate_class_weights(pdb_protBERT_dataset_train, device)
save_alpha_weights(alpha_train, CLASS_WEIGHT_PATH)
alpha = load_alpha_weights(CLASS_WEIGHT_PATH)
print(f"Alpha weights:{alpha}")

criterion = FocalLoss(alpha=alpha, gamma=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# Data loaders
train_loader = DataLoader(pdb_protBERT_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(pdb_protBERT_dataset_valid, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(pdb_protBERT_dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# Training loop with validation
best_valid_fmax = 0
no_improvement = 0

for epoch in range(1, EPOCHS + 1):
    # Training phase
    model.train()
    optimizer.zero_grad()
    
    for i, data in enumerate(train_loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.float())
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Validation phase
    valid_metrics = evaluate(valid_loader, model, device)
    train_metrics = evaluate(train_loader, model, device)
    
    # Update scheduler
    scheduler.step(valid_metrics['micro_aupr'])
    
    # Early stopping check
    if valid_metrics['fmax'] > best_valid_fmax:
        best_valid_fmax = valid_metrics['fmax']
        no_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved with Validation Fmax: {best_valid_fmax:.4f}")
    else:
        no_improvement += 1
    
    # Print metrics
    print(f'\nEpoch {epoch:03d}')
    print(f"Train Micro AUPR: {train_metrics['micro_aupr']:.4f} | Macro AUPR: {train_metrics['macro_aupr']:.4f}")
    print(f"Valid Micro AUPR: {valid_metrics['micro_aupr']:.4f} | Macro AUPR: {valid_metrics['macro_aupr']:.4f}")
    print(f"Valid Fmax: {valid_metrics['fmax']:.4f} (Best: {best_valid_fmax:.4f})")
    
    # Early stopping
    if no_improvement >= PATIENCE:
        print(f"No improvement for {PATIENCE} epochs. Stopping training.")
        break

# Final evaluation on test set
model.load_state_dict(torch.load(BEST_MODEL_PATH))
test_metrics = evaluate(test_loader, model, device)

print("\nFinal Test Results:")
print(f"Micro AUPR: {test_metrics['micro_aupr']:.4f}")
print(f"Macro AUPR: {test_metrics['macro_aupr']:.4f}")
print(f"Fmax: {test_metrics['fmax']:.4f} (Threshold: {test_metrics['best_threshold']:.2f})")

# Save model info
model_info = {
    "input_size": input_size,
    "hidden_sizes": hidden_sizes,
    "output_size": output_size
}
with open(MODEL_INFO_PATH, 'w') as f:
    json.dump(model_info, f)