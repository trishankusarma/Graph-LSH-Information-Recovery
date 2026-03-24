# Stacks everything, L layers
"""
transformer_model.py
────────────────────
Full model: L layers of [LocalGAT + LSHAttention + GatedFusion + Recovery] + FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat_local      import LocalGATBranch
from .lsh_attention  import LearnedLSHAttention
from .fusion         import GatedFusion
from .recovery       import InformationRecovery


class TransformerLayer(nn.Module):
    """Single layer of the sparse graph transformer."""

    def __init__(self, config):
        super().__init__()

        # Q, K, V projections  (input is z = [x || λ] at layer 0, else h)
        self.W_Q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_K = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_V = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Two branches
        self.local_branch  = LocalGATBranch(config)
        self.global_branch = LearnedLSHAttention(config)

        # Fusion + recovery
        self.fusion        = GatedFusion(config)
        self.use_recovery  = config.use_recovery
        if use_recovery:
            self.recovery = InformationRecovery(config)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, h: torch.Tensor, lap_pe: torch.Tensor,
                edge_index: torch.Tensor, deg: torch.Tensor):
        """
        Returns
        -------
        h_out         : (N, d)
        bucket_logits : dict  'q', 'k'  → (N, B)
        confidence    : (N,)  or None
        V             : (N, d) value matrix (for recovery loss)
        """
        # Projections
        Q = self.W_Q(h)
        K = self.W_K(h)
        V = self.W_V(h)

        # Local branch
        h_local  = self.local_branch(h, edge_index)

        # Global branch (LSH attention)
        h_global, bucket_logits = self.global_branch(
            Q, K, V, lap_pe, edge_index, deg
        )

        # Fusion
        h_fused = self.fusion(h_local, h_global)
        h_fused = self.norm1(h + h_fused)            # residual

        # Recovery
        confidence = None
        if self.use_recovery:
            bk = bucket_logits["k"].argmax(dim=-1)
            h_fused, confidence = self.recovery(
                h_fused, V, bucket_logits["q"], bk
            )

        # FFN
        h_out = self.norm2(h_fused + self.ffn(h_fused))

        return h_out, bucket_logits, confidence, V


# ─────────────────────────────────────────────────────────────────────────────

class SparseGraphTransformer(nn.Module):
    """
    Full L-layer Sparse Graph Transformer with LSH Attention + Recovery.

    Args
    ----
    config : ModelConfig  from hyperparameters/config.py
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d   = config.hidden_dim
        k   = config.lap_dim

        # Input projection: (in_dim + lap_dim) → hidden_dim
        self.input_proj = nn.Linear(config.in_dim + k, d)

        # L transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(config.num_layers)
        ])

        # Output classifier
        self.classifier = nn.Linear(d, config.out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, lap_pe: torch.Tensor,
                edge_index: torch.Tensor, deg: torch.Tensor):
        """
        Returns
        -------
        logits         : (N, num_classes)
        aux            : dict with 'bucket_logits', 'confidences', 'Vs'
                         — used by loss functions
        """
        # Augment input with Laplacian PE then project
        z = torch.cat([x, lap_pe], dim=-1)           # (N, in_dim+k)
        h = self.input_proj(z)                        # (N, d)

        all_bucket_logits = []
        all_confidences   = []
        all_Vs            = []

        for layer in self.layers:
            h, bucket_logits, confidence, V = layer(
                h, lap_pe, edge_index, deg
            )
            all_bucket_logits.append(bucket_logits)
            all_confidences.append(confidence)
            all_Vs.append(V)

        logits = self.classifier(h)                  # (N, out_dim)

        aux = {
            "bucket_logits": all_bucket_logits,       # list of dicts per layer
            "confidences"  : all_confidences,         # list of (N,) or None
            "Vs"           : all_Vs,                  # list of (N, d)
        }
        return logits, aux