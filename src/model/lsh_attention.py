"""
lsh_attention.py
────────────────
Global Branch: Learned LSH Sparse Attention

Steps (from paper):
  1. Learned bucket assignment  via MLP on [Q || λ]
  2. Sparse pair construction   P = P_LSH ∪ P_graph
  3. Attention scores           s_ij with SPD + degree biases
  4. Sparse softmax             per query node
  5. Value aggregation          h_global
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


class LearnedLSHAttention(nn.Module):
    """
    Learned Graph-Aware LSH Attention (ε-LHA).

    Args
    ----
    hidden_dim   : d  — embedding dimension (must be divisible by num_heads)
    lap_dim      : k  — Laplacian PE dimension
    num_buckets  : B  — number of hash buckets
    num_heads    : H  — multi-head attention
    max_spd      : max shortest-path distance (for SPD bias embedding)
    max_degree   : max node degree (for degree bias embedding)
    dropout      : attention dropout
    """

    def __init__(self, config):
        super().__init__()

        assert config.hidden_dim % config.num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim  = config.hidden_dim
        self.lap_dim     = config.lap_dim
        self.num_buckets = config.num_buckets
        self.num_heads   = config.num_heads
        self.head_dim    = config.hidden_dim // config.num_heads
        self.max_spd     = config.max_spd
        self.dropout     = config.dropout

        # ── Bucket MLP for Queries ────────────────────────────────────
        # input: [Q || λ]  →  (hidden_dim + lap_dim)
        self.mlp_q = nn.Sequential(
            nn.Linear(config.hidden_dim + config.lap_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_buckets),
        )

        # ── Bucket MLP for Keys ───────────────────────────────────────
        self.mlp_k = nn.Sequential(
            nn.Linear(config.hidden_dim + config.lap_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_buckets),
        )

        # ── Structural bias embeddings ────────────────────────────────
        # SPD bias: one scalar per (head, spd_value)
        self.spd_bias    = nn.Embedding(config.num_spd_bins + 1, config.num_heads)
        # learnable bin boundaries
        self.register_buffer('spd_boundaries', torch.linspace(0, 2, config.num_spd_bins))   # PE distances ∈ [0, 2] roughly
        # Degree bias: src and dst separately
        self.deg_src_emb = nn.Embedding(config.max_degree + 2, config.num_heads)   # +2 for OOV
        self.deg_dst_emb = nn.Embedding(config.max_degree + 2, config.num_heads)

        self.attn_drop   = nn.Dropout(config.dropout)
        self.out_proj    = nn.Linear(config.hidden_dim, config.hidden_dim)

    # ─────────────────────────────────────────────────────────────────

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                lap_pe: torch.Tensor, edge_index: torch.Tensor,
                deg: torch.Tensor):
        """
        Parameters
        ----------
        Q, K, V      : (N, hidden_dim)
        lap_pe       : (N, lap_dim)
        edge_index   : (2, E)  — graph edges
        spd          : (N, N)  — clipped shortest-path distances (long)
        deg          : (N,)    — node degrees (long)

        Returns
        -------
        h_global     : (N, hidden_dim)
        bucket_logits: dict with 'q' and 'k' → (N, B)  [for hash loss]
        """
        N = Q.size(0)
        device = Q.device

        # ── Step 1: Bucket logits ─────────────────────────────────────
        q_input = torch.cat([Q, lap_pe], dim=-1)    # (N, d+k)
        k_input = torch.cat([K, lap_pe], dim=-1)

        l_q = self.mlp_q(q_input)                   # (N, B)
        l_k = self.mlp_k(k_input)                   # (N, B)

        bq = l_q.argmax(dim=-1)                     # (N,)  — query bucket ids
        bk = l_k.argmax(dim=-1)                     # (N,)  — key   bucket ids

        # ── Step 2: Sparse pair construction ─────────────────────────
        lsh_pairs  = self._lsh_pairs(bq, bk, N, device)    # (2, P_lsh)
        graph_pairs = edge_index                            # (2, E)

        # Union — concatenate then deduplicate
        all_pairs = torch.cat([lsh_pairs, graph_pairs], dim=1)  # (2, P)
        all_pairs = torch.unique(all_pairs, dim=1)               # deduplicate

        src_p = all_pairs[0]   # query nodes  i
        dst_p = all_pairs[1]   # key   nodes  j

        # ── Step 3: Attention scores ──────────────────────────────────
        # Reshape to multi-head
        Q_h = Q.view(N, self.num_heads, self.head_dim)   # (N, H, d/H)
        K_h = K.view(N, self.num_heads, self.head_dim)
        V_h = V.view(N, self.num_heads, self.head_dim)

        # Dot-product score per head
        q_pairs = Q_h[src_p]   # (P, H, d/H)
        k_pairs = K_h[dst_p]   # (P, H, d/H)
        scores  = (q_pairs * k_pairs).sum(-1) / (self.head_dim ** 0.5)  # (P, H)

        # SPD bias
        pe_i      = lap_pe[src_p]                         # (|P|, k)
        pe_j      = lap_pe[dst_p]                         # (|P|, k)
        pe_dist   = (pe_i - pe_j).norm(dim=-1)            # (|P|,)
        spd_proxy = torch.bucketize(pe_dist, self.spd_boundaries)  # (|P|,) long
        spd_b     = self.spd_bias(spd_proxy)              # (|P|, H)
        scores    = scores + spd_b

        # Degree bias
        deg_src   = deg[src_p].clamp(0, self.deg_src_emb.num_embeddings - 1)
        deg_dst   = deg[dst_p].clamp(0, self.deg_dst_emb.num_embeddings - 1)
        scores    = scores + self.deg_src_emb(deg_src) + self.deg_dst_emb(deg_dst)

        # ── Step 4: Sparse softmax (per query node, per head) ─────────
        # pyg softmax: normalises over all j for each i
        # We do it head-by-head stacked on dim 1
        alpha = pyg_softmax(scores, src_p, num_nodes=N)  # (P, H)
        alpha = self.attn_drop(alpha)

        # ── Step 5: Value aggregation ─────────────────────────────────
        # h_global_i = Σ_{j∈P(i)} α_ij · V_j
        v_pairs   = V_h[dst_p]                           # (P, H, d/H)
        weighted  = alpha.unsqueeze(-1) * v_pairs        # (P, H, d/H)

        h_global  = torch.zeros(N, self.num_heads, self.head_dim, device=device)
        h_global.scatter_add_(0,
            src_p.unsqueeze(-1).unsqueeze(-1).expand_as(weighted),
            weighted)

        h_global  = h_global.view(N, self.hidden_dim)   # (N, d)
        h_global  = self.out_proj(h_global)

        bucket_logits = {"q": l_q, "k": l_k}
        return h_global, bucket_logits

    # ─────────────────────────────────────────────────────────────────

    def _lsh_pairs(self, bq: torch.Tensor, bk: torch.Tensor,
                   N: int, device: torch.device) -> torch.Tensor:
        """
        Build P_LSH = {(i,j) | bQ(i) == bK(j)} efficiently.
        Groups nodes by bucket, then forms all pairs within the same bucket.
        """
        pairs = []
        for b in range(self.num_buckets):
            q_nodes = (bq == b).nonzero(as_tuple=True)[0]   # nodes with bQ=b
            k_nodes = (bk == b).nonzero(as_tuple=True)[0]   # nodes with bK=b
            if q_nodes.numel() == 0 or k_nodes.numel() == 0:
                continue
            # Cartesian product
            ii = q_nodes.unsqueeze(1).expand(-1, k_nodes.size(0)).reshape(-1)
            jj = k_nodes.unsqueeze(0).expand(q_nodes.size(0), -1).reshape(-1)
            # Remove self-loops
            mask = ii != jj
            pairs.append(torch.stack([ii[mask], jj[mask]], dim=0))

        if not pairs:
            # Fallback: no LSH pairs found, return empty
            return torch.zeros(2, 0, dtype=torch.long, device=device)

        return torch.cat(pairs, dim=1)                       # (2, P_lsh)