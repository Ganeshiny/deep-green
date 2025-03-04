import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, 
    GINConv,
    GlobalAttention,
    LayerNorm
)
from torch_geometric.utils import dropout_edge  # Updated function

class HierarchicalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_heads=4, dropout=0.3, go_hierarchy=None):
        super().__init__()
        self.go_hierarchy = go_hierarchy  # GO hierarchy adjacency matrix
        
        # Feature Fusion Layer
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout)
        )
        
        # Build Triplets of Layers (GATv2Conv, GINConv, LayerNorm)
        self.triplet_layers = nn.ModuleList()
        num_triplets = len(hidden_dims) - 1
        for i in range(num_triplets):
            gat = GATv2Conv(hidden_dims[i], hidden_dims[i] // num_heads, heads=num_heads)
            gin = GINConv(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dims[i+1])
                ))
            norm = LayerNorm(hidden_dims[i+1])
            self.triplet_layers.append(nn.ModuleList([gat, gin, norm]))
        
        # Hierarchical Attention Pooling
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.Tanh(),
                nn.Linear(hidden_dims[-1], 1)
            )
        )
        
        # Label-aware Prediction Heads
        self.label_heads = nn.ModuleList([
            LabelAwareHead(hidden_dims[-1], 256, output_dim, 
                           hierarchy=go_hierarchy[i] if go_hierarchy is not None else None)
            for i in range(output_dim)
        ])
        
        # Residual Connections for each triplet
        self.res_linear = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i+1])
            for i in range(num_triplets)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        # Feature Projection
        x = self.feature_proj(x)
        
        num_triplets = len(self.res_linear)
        for t in range(num_triplets):
            # Store input for residual connection
            x_res_in = x.clone()
            
            # GATv2Conv with dropout_edge
            gat_layer = self.triplet_layers[t][0]
            edge_index_dropped, _ = dropout_edge(edge_index, p=0.2, training=self.training)
            x = gat_layer(x, edge_index_dropped)
            
            # GINConv (pass edge_index)
            gin_layer = self.triplet_layers[t][1]
            x = gin_layer(x, edge_index)
            
            # LayerNorm
            norm_layer = self.triplet_layers[t][2]
            x = norm_layer(x)
            
            # Add residual connection (transform the stored input)
            x = x + self.res_linear[t](x_res_in)
            
            # Activation and dropout
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)
        
        # Hierarchical Pooling and Label-aware Prediction Heads
        graph_emb = self.pool(x, batch)
        outputs = [head(graph_emb) for head in self.label_heads]
        return torch.stack(outputs, dim=1)

class LabelAwareHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hierarchy=None):
        super().__init__()
        self.hierarchy = hierarchy  # Parent-child relationships
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=4, 
            kdim=input_dim, vdim=input_dim
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # Attend to hierarchy-relevant features if provided
        if self.hierarchy is not None:
            x, _ = self.attention(x, x, x)
        return self.mlp(x)
