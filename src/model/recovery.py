# Information Recovery Module
"""
recovery.py
───────────
Information Recovery Module 

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
    config.hidden_dim : d  — embedding dimension
    config.num_buckets: B
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim  = config.hidden_dim
        self.num_buckets = config.num_buckets

        # Projects prototype-weighted sum back to hidden_dim
        self.W_r = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        # No norm here — TransformerLayer handles LayerNorm after recovery

    def forward(self, h_fused: torch.Tensor, V: torch.Tensor,
                bucket_logits_q: torch.Tensor,
                bucket_logits_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_fused          : (N, d)   — output of gated fusion
        V                : (N, d)   — value matrix
        bucket_logits_q  : (N, B)   — ℓ_Q (raw, before softmax)
        bk               : (N,)     — hard key-bucket assignment (long)

        Returns
        -------
        h_post_recovery : (N, d)  — recovered embeddings
        confidence      : (N,)   — per-node confidence score ∈ [0,1]
        """
        bk = bucket_logits_k.argmax(dim=-1)

        N, d = h_fused.shape
        B    = self.num_buckets
        device = h_fused.device

        # ── Soft bucket weights ───────────────────────────────────────
        p_q = F.softmax(bucket_logits_q, dim=-1)       # (N, B)

        # ── Bucket prototypes — fully vectorised ──────────────────────
        # μ_b = mean(V_j | bK(j) = b)
        # scatter_add to sum V into each bucket, divide by count
        counts     = torch.zeros(B, device=device)
        prototypes = torch.zeros(B, d, device=device)

        counts.scatter_add_(
            0, bk,
            torch.ones(N, device=device)
        )                                               # (B,) — nodes per bucket

        prototypes.scatter_add_(
            0,
            bk.unsqueeze(-1).expand(-1, d),            # (N, d) index
            V
        )                                               # (B, d) — sum of V per bucket

        # Mean — clamp counts to avoid div by zero
        prototypes = prototypes / counts.clamp(min=1).unsqueeze(-1)   # (B, d)

        # Fallback for empty buckets → global mean of V
        empty = counts == 0
        if empty.any():
            prototypes[empty] = V.mean(dim=0)

        # ── Residual  r_i = W_r · Σ_b p_b^(i) · μ_b ─────────────────
        proto_weighted = p_q @ prototypes              # (N, d)
        residual       = self.W_r(proto_weighted)      # (N, d)

        # ── Confidence  c_i = 1 - H(p) / log(B) ─────────────────────
        # H(p) = -Σ p_b log(p_b)  — clamp for numerical stability
        entropy     = -(p_q * p_q.clamp(min=1e-9).log()).sum(dim=-1)  # (N,)
        max_entropy = torch.log(torch.tensor(float(B), device=device))
        confidence  = 1.0 - entropy / max_entropy                      # (N,) ∈ [0,1]

        # ── Confidence-gated recovery ─────────────────────────────────
        # uncertain nodes (low c) receive more residual signal
        gate = (1.0 - confidence).unsqueeze(-1)             # (N, 1)
        h_post_recovery = h_fused + gate * residual                    # (N, d)

        return h_post_recovery, confidence