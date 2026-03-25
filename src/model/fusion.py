# Gated local-global fusion
"""
fusion.py
─────────
Gated Local–Global Fusion

  g_i = σ( W_g · [h_local_i || h_global_i] )
  h_i = g_i ⊙ h_local_i  +  (1 - g_i) ⊙ h_global_i
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Learnable gate that blends local (GAT) and global (LSH) representations.

    Args
    ----
    hidden_dim : d
    """

    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(2 * config.hidden_dim, config.hidden_dim)

    def forward(self, h_local: torch.Tensor, h_global: torch.Tensor) -> torch.Tensor:
        """
        h_local, h_global : (N, d)
        returns            : (N, d)
        """
        g = torch.sigmoid(self.gate(torch.cat([h_local, h_global], dim=-1)))  # (N, d)
        return g * h_local + (1 - g) * h_global