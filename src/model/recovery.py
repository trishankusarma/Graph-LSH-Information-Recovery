# Information Recovery Module
"""
recovery.py
───────────
Information Recovery Module  ← Novel Contribution

INTUITION
─────────
Sparse attention (P ⊂ N²) drops pairs (i,j) ∉ P.
Dropped pairs may carry task-relevant signal.
We estimate what was missed using:
  • Bucket prototypes  μ_b = mean(V_j for j in bucket_b)
  • Soft bucket weights p_b^(i) = softmax(ℓ_Q^(i))
  • Confidence score   c_i = 1 - entropy(p^(i)) / log(B)
                            → high c_i = node is confidently bucketed
                            → low  c_i = node is uncertain = more info dropped

Recovery residual:
  r_i = W_r · Σ_b  p_b^(i) · μ_b

Confidence-gated output:
  h_i^rec = h_i + (1 - c_i) · r_i
           ↑ uncertain nodes get more recovery signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InformationRecovery(nn.Module):
    """
    Args
    ----
    hidden_dim   : d  — embedding dimension
    num_buckets  : B
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim  = config.hidden_dim
        self.num_buckets = config.num_buckets

        # Projects prototype-weighted sum back to hidden_dim
        self.W_r   = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.norm  = nn.LayerNorm(config.hidden_dim)

    # ─────────────────────────────────────────────────────────────────

    def forward(self, h_fused: torch.Tensor, V: torch.Tensor,
                bucket_logits_q: torch.Tensor,
                bk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_fused          : (N, d)   — output of gated fusion
        V                : (N, d)   — value matrix (same as attention V)
        bucket_logits_q  : (N, B)   — ℓ_Q (raw, before softmax)
        bk               : (N,)     — hard key-bucket assignment (long)

        Returns
        -------
        h_recovered   : (N, d)  — recovered embeddings
        confidence    : (N,)    — per-node confidence score ∈ [0,1]
        """
        N, d = h_fused.shape
        B    = self.num_buckets
        device = h_fused.device

        # ── Soft bucket weights ───────────────────────────────────────
        p_q = F.softmax(bucket_logits_q, dim=-1)       # (N, B)

        # ── Bucket prototypes  μ_b = mean(V_j | bK(j) = b) ───────────
        prototypes = torch.zeros(B, d, device=device)
        counts     = torch.zeros(B, device=device)

        for b in range(B):
            mask = bk == b
            if mask.any():
                prototypes[b] = V[mask].mean(dim=0)
                counts[b]     = mask.sum().float()

        # Fallback for empty buckets: use global mean
        empty = counts == 0
        if empty.any():
            global_mean = V.mean(dim=0)
            prototypes[empty] = global_mean

        # ── Residual  r_i = W_r · Σ_b p_b^(i) · μ_b ─────────────────
        # p_q: (N, B),  prototypes: (B, d)  →  (N, d)
        proto_weighted = p_q @ prototypes              # (N, d)
        residual       = self.W_r(proto_weighted)      # (N, d)

        # ── Confidence  c_i = 1 - H(p) / log(B) ─────────────────────
        #   H(p) = -Σ p_b log(p_b)   ∈ [0, log(B)]
        entropy    = -(p_q * (p_q + 1e-9).log()).sum(dim=-1)   # (N,)
        max_entropy = torch.log(torch.tensor(float(B), device=device))
        confidence  = 1.0 - entropy / max_entropy               # (N,)  ∈ [0,1]

        # ── Confidence-gated recovery ─────────────────────────────────
        #   Uncertain nodes (low c) receive more recovery
        gate = (1.0 - confidence).unsqueeze(-1)                 # (N, 1)
        h_recovered = h_fused + gate * residual                 # (N, d)
        h_recovered = self.norm(h_recovered)

        return h_recovered, confidence