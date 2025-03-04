import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

# Load precomputed predictions and labels
with open('/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/GCN-PDB_BP_results.pckl', 'rb') as f:
    data = pickle.load(f)
    y_true, y_pred = data['Y_true'], data['Y_pred']

# Ensure proper shape
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Function to compute Fmax across multiple thresholds
def compute_fmax(y_true, y_pred):
    thresholds = np.linspace(0, 1, 101)
    fmax_values = []
    
    for threshold in thresholds:
        y_bin = (y_pred >= threshold).astype(int)
        f1_micro = f1_score(y_true, y_bin, average='micro')
        f1_macro = f1_score(y_true, y_bin, average='macro')
        fmax_values.append((threshold, f1_micro, f1_macro))
    
    best_micro = max(fmax_values, key=lambda x: x[1])
    best_macro = max(fmax_values, key=lambda x: x[2])
    
    return best_micro, best_macro, fmax_values

# Compute AUPR dynamically
def compute_aupr(y_true, y_pred):
    aupr_micro = average_precision_score(y_true, y_pred, average='micro')
    aupr_macro = average_precision_score(y_true, y_pred, average='macro')
    return aupr_micro, aupr_macro

# Get best Fmax and AUPR
best_micro, best_macro, fmax_values = compute_fmax(y_true, y_pred)
aupr_micro, aupr_macro = compute_aupr(y_true, y_pred)

# Print results
print(f'Best Fmax (Micro): {best_micro[1]:.4f} at threshold {best_micro[0]:.2f}')
print(f'Best Fmax (Macro): {best_macro[2]:.4f} at threshold {best_macro[0]:.2f}')
print(f'AUPR (Micro): {aupr_micro:.4f}')
print(f'AUPR (Macro): {aupr_macro:.4f}')

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'PR Curve (AUPR: {aupr_micro:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Plot Fmax vs Threshold
thresholds = [t[0] for t in fmax_values]
f1_micro_values = [t[1] for t in fmax_values]
f1_macro_values = [t[2] for t in fmax_values]

plt.figure(figsize=(7, 5))
plt.plot(thresholds, f1_micro_values, label='F1 Micro', marker='o')
plt.plot(thresholds, f1_macro_values, label='F1 Macro', marker='s')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('Fmax vs Threshold')
plt.legend()
plt.show()
