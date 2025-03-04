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

class GCN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GCN, self).__init__()
        
        # Feature Transformation
        self.initial_bn = nn.BatchNorm1d(input_size)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)  # Reduce dropout
        )
        # Graph Convolution Layers (SAGE + GATv2)
        self.conv_layers = nn.ModuleList([
            GCNConv(hidden_sizes[0], hidden_sizes[1]),  # Use GCN instead of SAGE
            GATv2Conv(hidden_sizes[1], hidden_sizes[1]//4, heads=4)
        ])
        self.bn_layers = nn.ModuleList([
            nn.LayerNorm(hidden_sizes[1]),  # LayerNorm instead of BatchNorm
            nn.LayerNorm(hidden_sizes[1])
        ])
        # Hierarchical Pooling
        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_sizes[1], hidden_sizes[1]),
                nn.Tanh(),
                nn.Linear(hidden_sizes[1], 1)
            )
        )

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[1] + input_size, output_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(output_size),
            nn.Dropout(0.2)  # Helps prevent overfitting
        )
        # Regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = self.initial_bn(x)
        x_res = self.input_proj(x)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.leaky_relu(bn(conv(x, edge_index)), 0.2)
            x = self.dropout(x)

            if self.training:
                edge_index, _ = dropout_edge(edge_index, p=0.2)

        graph_emb = self.pool(x, batch)
        global_res = global_mean_pool(x_res, batch)
        final_emb = torch.cat([graph_emb, global_res], dim=1)

        return self.output_layer(final_emb)



class GCN2(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GCN2, self).__init__()
        
        # Feature Transformation
        self.initial_bn = nn.BatchNorm1d(input_size)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )

        # Graph Convolution Layers (SAGE + GATv2)
        self.conv_layers = nn.ModuleList([
            SAGEConv(hidden_sizes[0], hidden_sizes[1]),
            GATv2Conv(hidden_sizes[1], hidden_sizes[1]//4, heads=4)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1])
        ])

        # Hierarchical Pooling
        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_sizes[1], hidden_sizes[1]),
                nn.Tanh(),
                nn.Linear(hidden_sizes[1], 1)
            )
        )

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[1] + input_size, output_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(output_size)
        )

        # Regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = self.initial_bn(x)
        x_res = self.input_proj(x)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.leaky_relu(bn(conv(x, edge_index)), 0.2)
            x = self.dropout(x)

            if self.training:
                edge_index, _ = dropout_edge(edge_index, p=0.2)

        graph_emb = self.pool(x, batch)
        global_res = global_mean_pool(x_res, batch)
        final_emb = torch.cat([graph_emb, global_res], dim=1)

        return self.output_layer(final_emb)