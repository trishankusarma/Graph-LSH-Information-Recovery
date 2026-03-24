# KL(ψ̃ || p_Q) from LHA paper
"""
hash_loss.py
────────────
Hash Supervision Loss (from paper Section 1.1)

Structural utility for query i and bucket b:
  ψ_b^(i) = Σ_{j ∈ bucket_b}  A_ij · g(SPD_ij, λ_i, λ_j)
  g        = 1 + γ·exp(-SPD_ij) + δ·cos(λ_i, λ_j)

Target:
  ψ̃^(i) = ψ^(i) / Σ_b ψ_b^(i)

Loss:
  L_hash = KL(ψ̃^(i) || p_Q^(i))  +  KL(ψ̃'^(j) || p_K^(j))
"""

import torch
import torch.nn.functional as F


def structural_utility(adj: torch.Tensor,
                        lap_pe: torch.Tensor, bk: torch.Tensor,
                        num_buckets: int, delta: float = 0.5) -> torch.Tensor:
    """
    Compute ψ̃^(i) — normalised structural utility per node per bucket.

    Parameters
    ----------
    adj          : (N, N) float  — adjacency matrix  (can be sparse approx)
    lap_pe       : (N, k) float  — Laplacian PE
    bk           : (N,)   long   — hard key-bucket assignment
    num_buckets  : B
    gamma, delta : structural utility coefficients

    Returns
    -------
    psi_norm : (N, B) — target distribution over buckets
    """
    N = adj.size(0)
    device = adj.device

    # Cosine similarity between PE vectors  (N, N)
    lap_norm = F.normalize(lap_pe, p=2, dim=-1)
    cos_sim  = lap_norm @ lap_norm.T                              # (N, N)

    # g(SPD_ij, λ_i, λ_j) = 1 + γ·exp(-SPD) + δ·cos(λ_i,λ_j)
    g = (1.0
        #  + gamma * torch.exp(-spd.float())
         + delta * cos_sim)                                       # (N, N)

    # A · g   elementwise
    ag = adj * g                                                  # (N, N)

    # ψ_b^(i) = Σ_{j ∈ bucket_b} A_ij · g_ij
    psi = torch.zeros(N, num_buckets, device=device)
    for b in range(num_buckets):
        mask = (bk == b)                                          # (N,)
        if mask.any():
            psi[:, b] = ag[:, mask].sum(dim=-1)

    # Normalise to get target distribution
    psi_sum  = psi.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    psi_norm = psi / psi_sum                                      # (N, B)
    return psi_norm


def hash_supervision_loss(bucket_logits: dict, adj: torch.Tensor, lap_pe: torch.Tensor,
                          num_buckets: int, delta: float = 0.5) -> torch.Tensor:
    """
    L_hash = KL(ψ̃ || p_Q) + KL(ψ̃' || p_K)

    Parameters
    ----------
    bucket_logits : dict with keys 'q' (N,B) and 'k' (N,B)
    adj           : (N, N) adjacency (float)
    lap_pe        : (N, k) float
    num_buckets   : B

    Returns
    -------
    loss : scalar tensor
    """
    l_q = bucket_logits["q"]   # (N, B)
    l_k = bucket_logits["k"]   # (N, B)

    bq  = l_q.argmax(dim=-1)   # (N,) hard assignments
    bk  = l_k.argmax(dim=-1)

    p_q = F.softmax(l_q, dim=-1)    # (N, B)
    p_k = F.softmax(l_k, dim=-1)

    # Target distributions
    psi_q = structural_utility(adj, lap_pe, bk, num_buckets, delta)
    psi_k = structural_utility(adj.T, lap_pe, bq, num_buckets, delta)

    # KL divergence: KL(target || predicted) = Σ target · log(target/pred)
    # F.kl_div expects log-probabilities for input
    loss_q = F.kl_div(p_q.log().clamp(min=-1e9), psi_q, reduction="batchmean")
    loss_k = F.kl_div(p_k.log().clamp(min=-1e9), psi_k, reduction="batchmean")

    return loss_q + loss_k