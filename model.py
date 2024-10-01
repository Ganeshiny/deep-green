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
        self.dropout_input = nn.Dropout(0.5)
        #self.dropout_hidden = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        # Linear layer - input features
        x = torch.relu(self.linear_input(x))
        x = self.dropout_input(x)

        # GCN layers with decreasing hidden sizes
        for conv_layer in self.conv_layers:
            x = torch.relu(conv_layer(x, edge_index))
            #x = self.dropout_hidden(x)

        # Aggregation step - since this is a graph level classification, the values in each layer must be made coarse to represent one graph
        x = global_mean_pool(x, batch)

        outputs = self.output_layers(x) #logits output 
        return outputs

