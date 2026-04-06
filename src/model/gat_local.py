# Local branch (standard GAT)
"""
gat_local.py
────────────
Local Branch: Graph Attention Network (GAT) over 1-hop neighbours.
Uses PyG's GATConv as the backbone.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class LocalGATBranch(nn.Module):
    """
    Local message-passing branch.

    Args
    ----
    hidden_dim : d  — input AND output embedding dimension
    num_heads  : H  — GAT attention heads (output is projected back to d)
    gat_dropout    : float
    """

    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0

        self.gat = GATConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim // config.num_heads,
            heads=config.num_heads,
            dropout=config.gat_dropout,
            concat=True,               # output = num_heads × (hidden_dim/num_heads) = hidden_dim
        )
        self.drop = nn.Dropout(config.gat_dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        normalized x          : (N, hidden_dim)
        edge_index : (2, E)
        returns    : (N, hidden_dim)
        """
        h = self.gat(x, edge_index)    # (N, hidden_dim) :: attend on normalised input
        return self.drop(h)