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
from preprocessing.create_batch_dataset import PDB_Dataset
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, AttentionalAggregation


class GCN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GCN, self).__init__()

        # Linear layer - input features
        self.linear_input = nn.Linear(input_size, hidden_sizes[0])

        # GCN layers with decreasing hidden sizes
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.conv_layers.append(GCNConv(hidden_sizes[i], hidden_sizes[i + 1]))

        # Output layers for molecular function ontology
        self.output_layers = nn.Linear(hidden_sizes[-1], output_size)

        # Dropout layers
        self.dropout_input = nn.Dropout(0.3)
        #self.dropout_hidden = nn.Dropout(0.3)

        self.leaky_relu = nn.LeakyReLU(0.4)

    def forward(self, x, edge_index, batch):
        # Linear layer - input features
        x = self.leaky_relu(self.linear_input(x))
        x = self.dropout_input(x)

        # GCN layers with decreasing hidden sizes
        for conv_layer in self.conv_layers:
            x = self.leaky_relu(conv_layer(x, edge_index))
            #x = self.dropout_hidden(x)

        # Aggregation step - since this is a graph level classification, the values in each layer must be made coarse to represent one graph
        x = global_mean_pool(x, batch)

        outputs = self.output_layers(x) #logits output 
        return outputs

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
        self.attention_pool = AttentionalAggregation(
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
