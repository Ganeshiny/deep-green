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
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GlobalAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, 
    GATConv, 
    GlobalAttention, 
    global_mean_pool, 
    BatchNorm
)
from torch_geometric.utils import dropout_adj

class GCN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GCN, self).__init__()
        
        # Initial feature transformation
        self.initial_ln = nn.LayerNorm(input_size)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )

        # Graph Convolution Layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            # Alternate between GCN and GAT
            if i % 2 == 0:
                conv = GCNConv(hidden_sizes[i], hidden_sizes[i+1])
            else:
                conv = GATConv(hidden_sizes[i], hidden_sizes[i+1]//4, heads=4)
                
            self.conv_layers.append(conv)
            self.bn_layers.append(BatchNorm(hidden_sizes[i+1]))

        # Hierarchical attention pooling
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
                nn.Tanh(),
                nn.Linear(hidden_sizes[-1], 1)
            )
        )

        # Output layer with residual connection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1] + input_size, output_size),
            nn.BatchNorm1d(output_size)
        )

        # Regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # Initial processing
        x_res = self.initial_ln(x)
        x = self.input_proj(x_res)

        # Graph convolution blocks
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.leaky_relu(bn(conv(x, edge_index)), 0.2)
            x = self.dropout(x)
            
            # Edge dropout for regularization
            if self.training:
                edge_index, _ = dropout_adj(edge_index, p=0.2)

        # Hierarchical pooling
        graph_emb = self.pool(x, batch)

        # Residual connection with initial features
        global_res = global_mean_pool(x_res, batch)  
        final_emb = torch.cat([graph_emb, global_res], dim=1)

        return self.output_layer(final_emb)  # Shape: (batch_size, num_classes)


class GCN2(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GCN2, self).__init__()
        
        # **BatchNorm for Stabilization**
        self.initial_bn = nn.BatchNorm1d(input_size)
        
        # **Feature Transformation**
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)  # Reduced dropout for better learning
        )

        # **Graph Convolution Layers**
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            # Alternate between GCN and GAT
            if i % 2 == 0:
                conv = GCNConv(hidden_sizes[i], hidden_sizes[i+1])
            else:
                conv = GATConv(hidden_sizes[i], hidden_sizes[i+1]//4, heads=4)
                
            self.conv_layers.append(conv)
            self.bn_layers.append(BatchNorm(hidden_sizes[i+1]))

        # **Hierarchical Attention Pooling**
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
                nn.Tanh(),
                nn.Linear(hidden_sizes[-1], 1)
            )
        )

        # **Final Output Layer with Residual Connection**
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1] + input_size, output_size),
            nn.LeakyReLU(0.2),  # Helps push logits apart before sigmoid
            nn.BatchNorm1d(output_size)
        )

        # **Regularization**
        self.dropout = nn.Dropout(0.2)  # Reduced dropout to avoid excessive feature loss

    def forward(self, x, edge_index, batch):
        # **Initial Processing with BatchNorm**
        x = self.initial_bn(x)
        x_res = x  # Residual connection
        x = self.input_proj(x)

        # **Graph Convolution Blocks**
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.leaky_relu(bn(conv(x, edge_index)), 0.2)
            x = self.dropout(x)
            
            # **Edge Dropout for Regularization**
            if self.training:
                edge_index, _ = dropout_adj(edge_index, p=0.2)

        # **Hierarchical Pooling**
        graph_emb = self.pool(x, batch)

        # **Residual Connection with Initial Features**
        global_res = global_mean_pool(x_res, batch)  
        final_emb = torch.cat([graph_emb, global_res], dim=1)

        return self.output_layer(final_emb)  # **1D label array output**


class RareLabelGNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_attention_heads=4):
        super(RareLabelGNN, self).__init__()

        # Input Linear Layer
        self.input_linear = nn.Linear(input_size, hidden_sizes[0])

        # GNN Layers (using GAT for attention-based learning)
        self.gnn_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.gnn_layers.append(
                GATConv(hidden_sizes[i], hidden_sizes[i + 1], heads=num_attention_heads, concat=False)
            )

        # Global Pooling (Attention-based pooling)
        self.attention_pool = GlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(hidden_sizes[-1], 128), nn.ReLU(), nn.Linear(128, 1))
        )

        # Output Layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x, edge_index, batch):
        # Input transformation
        x = F.leaky_relu(self.input_linear(x), negative_slope=0.1)

        # GNN layers
        for gnn_layer in self.gnn_layers:
            x = F.leaky_relu(gnn_layer(x, edge_index), negative_slope=0.1)

        # Global Pooling
        graph_embedding = self.attention_pool(x, batch)

        # Output Layer (Graph-Level Prediction)
        out = self.output_layer(graph_embedding)
        return out
