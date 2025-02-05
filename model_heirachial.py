import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, 
    GINConv,
    global_add_pool,
    GlobalAttention,
    LayerNorm,
    TransformerConv
)
from torch_geometric.utils import dropout_adj


#GIT!!!!!!!!!!!
class HierarchicalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_heads=8, dropout=0.3, go_hierarchy=None):
        super().__init__()
        self.go_hierarchy = go_hierarchy  # GO hierarchy adjacency matrix
        
        # Feature Fusion Layer
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout)
        )
        
        # Multimodal Graph Convolution Blocks
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.conv_layers.extend([
                GATv2Conv(hidden_dims[i], hidden_dims[i]//num_heads, heads=num_heads),
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dims[i+1])
                )),
                LayerNorm(hidden_dims[i+1])
            ])
        
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
                          hierarchy=go_hierarchy[i])
            for i in range(output_dim)
        ])
        
        # Residual Connections
        self.res_linear = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i+1])
            for i in range(len(hidden_dims)-1)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        # Feature Projection
        x = self.feature_proj(x)
        
        # Multimodal Convolution
        for i, layer in enumerate(self.conv_layers):
            if isinstance(layer, GATv2Conv):
                # Edge dropout for regularization
                edge_index, _ = dropout_adj(edge_index, p=0.2, 
                                          training=self.training)
                x = layer(x, edge_index)
            else:
                x_res = self.res_linear[i//3](x) if i%3 == 2 else 0
                x = layer(x) + x_res
                x = F.leaky_relu(x, 0.2)
                x = self.dropout(x)
        
        # Hierarchical Pooling
        graph_emb = self.pool(x, batch)
        
        # Label-specific Predictions
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
        # Attend to hierarchy-relevant features
        if self.hierarchy is not None:
            x, _ = self.attention(x, x, x)
        return self.mlp(x)
    

