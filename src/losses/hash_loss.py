# KL(ψ̃ || p_Q) from LHA paper
"""
hash_loss.py
────────────
Hash Supervision Loss (from paper Section 1.1)

Structural utility for query i and bucket b:
  ψ_b^(i) = Σ_{j ∈ bucket_b}  A_ij · g(pe_dist_ij, λ_i, λ_j)
  g        = 1 + γ·exp(-pe_dist_ij) + δ·cos(λ_i, λ_j)
  
  pe_dist_ij = ||λ_i - λ_j||₂  ← proxy for SPD, no BFS needed

Target:
  ψ̃^(i) = ψ^(i) / Σ_b ψ_b^(i)

Loss:
  L_hash = KL(ψ̃^(i) || p_Q^(i))  +  KL(ψ̃'^(j) || p_K^(j))
"""

import torch
import torch.nn.functional as F


def structural_utility(edge_index: torch.Tensor,
                       lap_pe: torch.Tensor,
                       bk: torch.Tensor,
                       num_nodes: int,
                       num_buckets: int,
                       gamma: float = 1.0,
                       delta: float = 0.5) -> torch.Tensor:
    """
    Compute ψ̃^(i) — normalised structural utility per node per bucket.
    Computed over edges only — never builds N×N matrix.

    Parameters
    ----------
    edge_index   : (2, E)  — graph edges
    lap_pe       : (N, k)  — Laplacian PE
    bk           : (N,)    — hard key-bucket assignment
    num_nodes    : N
    num_buckets  : B
    gamma, delta : structural utility coefficients

    Returns
    -------
    psi_norm : (N, B) — target distribution over buckets
    """
    device = lap_pe.device
    src, dst = edge_index[0], edge_index[1]    # (E,)

    # PE vectors for each edge endpoint
    pe_src = lap_pe[src]                        # (E, k)
    pe_dst = lap_pe[dst]                        # (E, k)

    # PE distance — proxy for SPD
    pe_dist  = (pe_src - pe_dst).norm(dim=-1)   # (E,)

    # Cosine similarity — only for edge pairs
    cos_sim  = F.cosine_similarity(pe_src, pe_dst, dim=-1)   # (E,)

    # g for each edge — same formula, SPD replaced by pe_dist
    g_edges  = 1.0 + gamma * torch.exp(-pe_dist) + delta * cos_sim   # (E,)

    # ψ_b^(i) = Σ_{j ∈ bucket_b, (i,j) ∈ E}  g_ij
    # For each edge (src→dst): if dst is in bucket b → add g to psi[src, b]
    psi = torch.zeros(num_nodes, num_buckets, device=device)
    for b in range(num_buckets):
        dst_in_bucket = (bk[dst] == b).float()          # (E,) — 1 if dst ∈ bucket b
        weighted      = g_edges * dst_in_bucket          # (E,) — zero out non-bucket edges
        psi[:, b].scatter_add_(0, src, weighted)

    # Normalise → target distribution
    psi_sum  = psi.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    psi_norm = psi / psi_sum                             # (N, B)
    return psi_norm


def hash_supervision_loss(bucket_logits: dict,
                          edge_index: torch.Tensor,
                          lap_pe: torch.Tensor,
                          num_nodes: int,
                          num_buckets: int,
                          gamma: float = 1.0,
                          delta: float = 0.5) -> torch.Tensor:
    """
    L_hash = KL(ψ̃ || p_Q) + KL(ψ̃' || p_K)

    Parameters
    ----------
    bucket_logits : dict — 'q': (N, B),  'k': (N, B)
    edge_index    : (2, E)
    lap_pe        : (N, k)
    num_nodes     : N
    num_buckets   : B
    gamma, delta  : structural utility coefficients

    Returns
    -------
    loss : scalar tensor
    """
    l_q = bucket_logits["q"]             # (N, B)
    l_k = bucket_logits["k"]             # (N, B)

    bq  = l_q.argmax(dim=-1)             # (N,)
    bk  = l_k.argmax(dim=-1)             # (N,)

    p_q = F.softmax(l_q, dim=-1)         # (N, B)
    p_k = F.softmax(l_k, dim=-1)         # (N, B)

    # Forward direction  i→j : bk drives which bucket j falls in
    psi_q = structural_utility(
        edge_index, lap_pe, bk, num_nodes, num_buckets, gamma, delta
    )

    # Reverse direction  j→i : flip edge_index, bq drives bucket
    edge_index_T = edge_index.flip(0)    # (2, E) — reversed edges
    psi_k = structural_utility(
        edge_index_T, lap_pe, bq, num_nodes, num_buckets, gamma, delta
    )

    # KL(target || predicted) — F.kl_div expects log-input
    loss_q = F.kl_div(p_q.log().clamp(min=-1e9), psi_q, reduction="batchmean")
    loss_k = F.kl_div(p_k.log().clamp(min=-1e9), psi_k, reduction="batchmean")

    return loss_q + loss_k