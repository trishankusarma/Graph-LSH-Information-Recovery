# reconstruction objective
"""
reconstruction_loss.py
──────────────────────
Recovery Loss

L_recovery = Σ_i  c_i · || h_recovered_i - h_full_rank_i ||²

where:
  h_full_rank = soft-weighted prototype sum using ALL buckets (top-K approx)
  c_i         = per-node confidence score from recovery module

High-confidence nodes:  loss ≈ 0   (they were well-attended)
Low-confidence nodes:   loss is high if recovery is poor
"""
import torch
import torch.nn.functional as F

def recovery_loss(h_recovered: torch.Tensor, V: torch.Tensor,
                  bucket_logits_q: torch.Tensor, confidence: torch.Tensor,
                  prototypes: torch.Tensor = None) -> torch.Tensor:
    """
    Parameters
    ----------
    h_recovered      : (N, d)  — output of recovery module
    V                : (N, d)  — value matrix
    bucket_logits_q  : (N, B)  — raw bucket logits for queries
    confidence       : (N,)    — per-node confidence ∈ [0,1]
    prototypes       : (B, d)  — bucket prototypes (optional; recomputed if None)

    Returns
    -------
    loss : scalar tensor
    """
    N, d = h_recovered.shape
    B    = bucket_logits_q.size(-1)
    device = h_recovered.device

    # Soft top-K approximation as reconstruction target
    p_soft  = F.softmax(bucket_logits_q, dim=-1)   # (N, B) — use all buckets softly

    if prototypes is None:
        # Compute bucket prototypes from V
        # Use soft assignment: μ_b = Σ_i p_b^(i) · V_i  (soft mean)
        proto = p_soft.T @ V                        # (B, d)
        denom = p_soft.sum(dim=0).unsqueeze(-1).clamp(min=1e-9)  # (B, 1)
        prototypes = proto / denom                  # (B, d)

    # Target: full soft-weighted reconstruction
    h_target = p_soft @ prototypes                  # (N, d)

    # Weighted MSE: uncertain nodes contribute more to loss signal
    # confidence is high → these nodes trust their sparse attention → weight low
    # We flip: weight = 1 - confidence so uncertain nodes drive recovery learning
    weight = (1.0 - confidence).unsqueeze(-1)       # (N, 1)
    diff   = (h_recovered - h_target) ** 2          # (N, d)
    loss   = (weight * diff).mean()

    return loss