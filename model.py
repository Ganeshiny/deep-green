import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, AttentionalAggregation, GATv2Conv, SAGEConv
import torch
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv, 
    GATConv, 
    AttentionalAggregation, 
    global_mean_pool, 
    BatchNorm
)
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import (
    SAGEConv, GATv2Conv, GINConv, 
    AttentionalAggregation, JumpingKnowledge,
    global_mean_pool, BatchNorm, LayerNorm
)
from torch_geometric.utils import dropout_edge

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool

class HybridGCNGAT(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, 
                 gcn_layers, gat_layers, dropout_rate=0.3):
        super(HybridGCNGAT, self).__init__()

        assert gcn_layers + gat_layers == len(hidden_sizes), \
            "Total layers must equal number of hidden sizes"

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(dropout_rate)
        )

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.layer_types = []

        for i in range(len(hidden_sizes) - 1):
            in_dim, out_dim = hidden_sizes[i], hidden_sizes[i + 1]
            if i < gcn_layers:
                self.layers.append(GCNConv(in_dim, out_dim))
                self.layer_types.append("gcn")
            else:
                self.layers.append(GATv2Conv(in_dim, out_dim // 4, heads=4, concat=True))
                self.layer_types.append("gat")
            self.layer_norms.append(nn.LayerNorm(out_dim))

        self.pooling = global_mean_pool

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_sizes[-1] // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[-1] // 2, output_size)
        )

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)

        for layer, norm, typ in zip(self.layers, self.layer_norms, self.layer_types):
            if typ == "gcn":
                x = F.leaky_relu(norm(layer(x, edge_index)), 0.2)
            elif typ == "gat":
                x = F.leaky_relu(norm(layer(x, edge_index)), 0.2)
            x = F.dropout(x, training=self.training)

        graph_emb = self.pooling(x, batch)
        return self.pred_head(graph_emb)


'''
            **Evaluation Results:**
 **Micro AUPR:** 0.2320
 **Macro AUPR:** 0.2770
 **Micro Fmax:** 0.3134 (Best T: 0.46)
 **Macro Fmax:** 0.2030 (Best T: 0.45)
'''
