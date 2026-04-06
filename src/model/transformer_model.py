# Stacks everything, L layers
"""
transformer_model.py
────────────────────
Full model: L layers of [LocalGAT + LSHAttention + GatedFusion + Recovery] + FFN

Normalisation strategy: Pre-LN throughout
  Pre-LN: h → norm → sublayer → add(h, out)
  More stable than Post-LN for deeper models, no warmup needed.
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

        # Q, K, V projections
        self.W_Q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_K = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_V = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Two branches
        self.local_branch  = LocalGATBranch(config)
        self.global_branch = LearnedLSHAttention(config)

        # Fusion + recovery
        self.fusion       = GatedFusion(config)
        self.use_recovery = config.use_recovery
        if self.use_recovery:
            self.recovery = InformationRecovery(config)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

        # Pre-LN norms — one per sublayer
        self.norm_attn   = nn.LayerNorm(config.hidden_dim)   # before attention branches
        self.norm_ffn    = nn.LayerNorm(config.hidden_dim)   # before FFN

    def forward(self, h: torch.Tensor, lap_pe: torch.Tensor,
                edge_index: torch.Tensor, deg: torch.Tensor):
        """
        Pre-LN forward:
          h → norm → [local | global] → fuse → residual
            → norm → recovery (optional) → residual
            → norm → FFN → residual

        Returns
        -------
        h_out           : (N, d)
        bucket_logits   : dict 'q', 'k' → (N, B)
        confidence      : (N,) or None
        V               : (N, d)
        h_post_recovery : (N, d)
        """
        # ── Attention sublayer — Pre-LN ───────────────────────────────
        h_norm = self.norm_attn(h)                   # normalise input first

        Q = self.W_Q(h_norm)
        K = self.W_K(h_norm)
        V = self.W_V(h_norm)

        # Local branch operates on normalised h
        h_local  = self.local_branch(h_norm, edge_index)

        # Global branch
        h_global, bucket_logits = self.global_branch(
            Q, K, V, lap_pe, edge_index, deg
        )

        # Fuse local + global then residual add
        h_fused = self.fusion(h_local, h_global)     # (N, d)
        h       = h + h_fused                        # residual — no norm here (Pre-LN)

        # ── Recovery sublayer ─────────────────────────────────────────
        h_post_recovery = h                          # default — identity
        confidence      = None

        if self.use_recovery:
            h_post_recovery, confidence = self.recovery(
                h, V, bucket_logits["q"], bucket_logits["k"]    # ← pass raw (N,B) logits
            )
            h = h_post_recovery           # effectively h = h_post_recovery
                                                     # written as residual for clarity

        # ── FFN sublayer — Pre-LN ─────────────────────────────────────
        h_out = h + self.ffn(self.norm_ffn(h))       # pre-norm then residual

        return h_out, bucket_logits, confidence, V, h_post_recovery


# ─────────────────────────────────────────────────────────────────────────────

class SparseGraphTransformer(nn.Module):
    """
    Full L-layer Sparse Graph Transformer with LSH Attention + Recovery.

    Args
    ----
    config : ModelConfig from hyperparameters/config.py
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        d = config.hidden_dim
        k = config.lap_dim

        # Input projection: (in_dim + lap_dim) → hidden_dim
        self.input_proj = nn.Linear(config.in_dim + k, d)

        # Final norm before classifier 
        self.final_norm = nn.LayerNorm(d)

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
        logits : (N, num_classes)
        aux    : dict with 'bucket_logits', 'confidences', 'Vs', 'h_recovered'
        """
        # Augment input with Laplacian PE then project
        z = torch.cat([x, lap_pe], dim=-1)    # (N, in_dim + k)
        h = self.input_proj(z)                 # (N, d)

        all_bucket_logits = []
        all_confidences   = []
        all_Vs            = []
        all_h_recovered   = []

        for layer in self.layers:
            h, bucket_logits, confidence, V, h_post_recovery = layer(
                h, lap_pe, edge_index, deg
            )
            all_bucket_logits.append(bucket_logits)
            all_confidences.append(confidence)
            all_Vs.append(V)
            all_h_recovered.append(h_post_recovery)

        # Final norm before classification (Pre-LN convention)
        h      = self.final_norm(h)
        logits = self.classifier(h)            # (N, out_dim)

        aux = {
            "bucket_logits": all_bucket_logits,
            "confidences"  : all_confidences,
            "Vs"           : all_Vs,
            "h_recovered"  : all_h_recovered,
        }
        return logits, aux