"""
lsh_attention.py
────────────────
Global Branch: Learned LSH Sparse Attention

Steps (from paper):
  1. Learned bucket assignment  via MLP on [Q || λ]
  2. Block-wise bucket attention (no cartesian product materialisation)
  3. Vectorised edge attention
  4. Two-pass merge via log-sum-exp (mathematically exact)
  5. Final projection

Memory strategy:
  - Buckets processed as dense (Nq × Nk) blocks — fast GEMM, no pair indices
  - Edge attention fully vectorised over E edges
  - Online softmax merge combines both — single unified normalisation
  - No inplace ops on gradient tensors — autograd safe

Correctness:
  - Equivalent to full sparse softmax over P = P_LSH ∪ P_graph
  - Isolated nodes (no bucket keys, no edges) → zero output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLSHAttention(nn.Module):
    """
    Learned Graph-Aware LSH Attention — Block-wise + Two-pass merge.

    Args
    ----
    config.hidden_dim  : d — embedding dimension (divisible by num_heads)
    config.lap_dim     : k — Laplacian PE dimension
    config.num_buckets : B — number of hash buckets
    config.num_heads   : H — attention heads
    config.max_degree  : max node degree for degree bias embedding
    config.num_spd_bins: bins for PE-distance structural proxy
    config.dropout     : attention dropout
    """

    def __init__(self, config):
        super().__init__()

        assert config.hidden_dim % config.num_heads == 0, \
            "hidden_dim must be divisible by num_heads"

        self.hidden_dim  = config.hidden_dim
        self.lap_dim     = config.lap_dim
        self.num_buckets = config.num_buckets
        self.num_heads   = config.num_heads
        self.head_dim    = config.hidden_dim // config.num_heads
        self.dropout     = config.dropout

        # ── Bucket MLPs ───────────────────────────────────────────────
        # Input: [Q || λ] → (hidden_dim + lap_dim) → B bucket logits
        self.mlp_q = nn.Sequential(
            nn.Linear(config.hidden_dim + config.lap_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_buckets),
        )
        self.mlp_k = nn.Sequential(
            nn.Linear(config.hidden_dim + config.lap_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_buckets),
        )

        # ── Structural bias embeddings ────────────────────────────────
        # PE-distance proxy for SPD — binned into num_spd_bins buckets
        self.spd_bias = nn.Embedding(config.num_spd_bins + 1, config.num_heads)
        self.register_buffer(
            'spd_boundaries',
            torch.linspace(0, 2, config.num_spd_bins)
        )

        # Degree bias — separate for src and dst
        self.deg_src_emb = nn.Embedding(config.max_degree + 2, config.num_heads)
        self.deg_dst_emb = nn.Embedding(config.max_degree + 2, config.num_heads)

        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    # ─────────────────────────────────────────────────────────────────

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                lap_pe: torch.Tensor, edge_index: torch.Tensor,
                deg: torch.Tensor):
        """
        Parameters
        ----------
        Q, K, V    : (N, hidden_dim)
        lap_pe     : (N, lap_dim)
        edge_index : (2, E)
        deg        : (N,) long — node degrees

        Returns
        -------
        h_global     : (N, hidden_dim)
        bucket_logits: dict 'q' → (N,B), 'k' → (N,B)
        """
        N, H, D_h = Q.size(0), self.num_heads, self.head_dim
        device     = Q.device
        max_deg    = self.deg_src_emb.num_embeddings - 1

        # ── Step 1: Bucket assignments ────────────────────────────────
        q_input = torch.cat([Q, lap_pe], dim=-1)    # (N, d+k)
        k_input = torch.cat([K, lap_pe], dim=-1)

        l_q = self.mlp_q(q_input)                   # (N, B)
        l_k = self.mlp_k(k_input)                   # (N, B)

        bq = l_q.argmax(dim=-1)                     # (N,) hard query-bucket ids
        bk = l_k.argmax(dim=-1)                     # (N,) hard key-bucket ids

        # Sort nodes by bucket so each bucket is a contiguous slice
        q_idx = torch.argsort(bq)                   # (N,) — sorted query indices
        k_idx = torch.argsort(bk)                   # (N,) — sorted key indices

        # Reorder tensors into bucket-contiguous order
        Q_s    = Q[q_idx];      K_s    = K[k_idx];      V_s    = V[k_idx]
        PE_q_s = lap_pe[q_idx]; PE_k_s = lap_pe[k_idx]
        deg_q_s = deg[q_idx];   deg_k_s = deg[k_idx]

        # Bucket offsets: where each bucket starts/ends in sorted arrays
        q_counts  = torch.bincount(bq, minlength=self.num_buckets)
        k_counts  = torch.bincount(bk, minlength=self.num_buckets)
        q_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                q_counts.cumsum(0)])
        k_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                k_counts.cumsum(0)])

        # ── Step 2: Block-wise Bucket Attention ───────────────────────
        # Collect per-bucket (m, d, o) — NO inplace ops on grad tensors
        m_b_list, d_b_list, o_b_list = [], [], []

        for b in range(self.num_buckets):
            s_q, e_q = q_offsets[b].item(), q_offsets[b + 1].item()
            s_k, e_k = k_offsets[b].item(), k_offsets[b + 1].item()
            Nq = e_q - s_q

            if Nq == 0:
                continue   # no queries in this bucket — skip entirely

            if s_k == e_k:
                # Queries exist but no keys → contribute -inf / zeros
                m_b_list.append(torch.full((Nq, H), float('-inf'), device=device))
                d_b_list.append(torch.zeros(Nq, H, device=device))
                o_b_list.append(torch.zeros(Nq, H, D_h, device=device))
                continue

            # Slice bucket tensors and reshape to (H, Nq/Nk, D_h)
            q_b = Q_s[s_q:e_q].view(-1, H, D_h).transpose(0, 1)   # (H, Nq, D_h)
            k_b = K_s[s_k:e_k].view(-1, H, D_h).transpose(0, 1)   # (H, Nk, D_h)
            v_b = V_s[s_k:e_k].view(-1, H, D_h).transpose(0, 1)   # (H, Nk, D_h)

            # Raw attention scores: (H, Nq, Nk)
            S_b = torch.matmul(q_b, k_b.transpose(-2, -1)) / (self.head_dim ** 0.5)

            # PE-distance bias: (Nq, Nk) → (Nq, Nk, H) → (H, Nq, Nk)
            diff    = PE_q_s[s_q:e_q].unsqueeze(1) - PE_k_s[s_k:e_k].unsqueeze(0)
            dist    = diff.norm(dim=-1)                               # (Nq, Nk)
            spd_idx = torch.bucketize(dist, self.spd_boundaries)
            b_spd   = self.spd_bias(spd_idx)                         # (Nq, Nk, H)

            # Degree bias: (Nq, H) + (Nk, H) → broadcast (Nq, Nk, H)
            b_deg = (self.deg_src_emb(deg_q_s[s_q:e_q].clamp(0, max_deg)).unsqueeze(1)
                   + self.deg_dst_emb(deg_k_s[s_k:e_k].clamp(0, max_deg)).unsqueeze(0))

            # Add biases: permute to (H, Nq, Nk) and add to S_b
            S_b = S_b + (b_spd + b_deg).permute(2, 0, 1)

            # Self-loop mask — node cannot attend to itself within bucket
            curr_q_idx = q_idx[s_q:e_q]
            curr_k_idx = k_idx[s_k:e_k]
            self_loop  = curr_q_idx.unsqueeze(1) == curr_k_idx.unsqueeze(0)  # (Nq, Nk)
            S_b = S_b.masked_fill(self_loop.unsqueeze(0).expand(H, -1, -1), float('-inf'))

            # Local softmax statistics for this bucket (no inplace ops)
            m_block     = S_b.max(dim=-1).values.clamp(min=-1e9)              # (H, Nq)
            scale_block = torch.exp((S_b - m_block.unsqueeze(-1)).clamp(min=-30, max=30))
            d_block     = scale_block.sum(dim=-1)                             # (H, Nq)
            o_block     = torch.matmul(scale_block, v_b)                      # (H, Nq, D_h)

            # Store transposed to (Nq, H) / (Nq, H, D_h) for later concat
            m_b_list.append(m_block.transpose(0, 1))
            d_b_list.append(d_block.transpose(0, 1))
            o_b_list.append(o_block.transpose(0, 1))

        # Concatenate all bucket outputs (sorted order) and unsort to original order
        m_bucket_s = torch.cat(m_b_list, dim=0)           # (N, H)
        d_bucket_s = torch.cat(d_b_list, dim=0)           # (N, H)
        o_bucket_s = torch.cat(o_b_list, dim=0)           # (N, H, D_h)

        rev_q_idx = torch.argsort(q_idx)                  # inverse permutation
        m_bucket  = m_bucket_s[rev_q_idx]                 # (N, H) — original order
        d_bucket  = d_bucket_s[rev_q_idx]
        o_bucket  = o_bucket_s[rev_q_idx]

        # ── Step 3: Vectorised Edge Attention ─────────────────────────
        src, dst = edge_index[0], edge_index[1]

        Q_h = Q.view(N, H, D_h)
        K_h = K.view(N, H, D_h)
        V_h = V.view(N, H, D_h)

        # Raw edge scores: (E, H)
        S_e = (Q_h[src] * K_h[dst]).sum(dim=-1) / (self.head_dim ** 0.5)

        # PE-distance bias for edges
        dist_e  = (lap_pe[src] - lap_pe[dst]).norm(dim=-1)              # (E,)
        b_spd_e = self.spd_bias(torch.bucketize(dist_e, self.spd_boundaries))  # (E, H)

        # Degree bias for edges
        b_deg_e = (self.deg_src_emb(deg[src].clamp(0, max_deg))
                 + self.deg_dst_emb(deg[dst].clamp(0, max_deg)))        # (E, H)

        S_e = S_e + b_spd_e + b_deg_e                                   # (E, H)

        # Vectorised scatter-max for per-node max score
        src_2d  = src.unsqueeze(1).expand(-1, H)                        # (E, H)
        m_edge  = torch.full((N, H), float('-inf'), device=device)
        m_edge.scatter_reduce_(0, src_2d, S_e.detach(), reduce='amax', include_self=True)

        # Exp-scaled scores and scatter-sum for d and o
        scale_e = torch.exp((S_e - m_edge[src]).clamp(min=-30, max=30)) # (E, H)

        d_edge  = torch.zeros(N, H, device=device)
        d_edge.scatter_add_(0, src_2d, scale_e)                         # (N, H)

        o_edge  = torch.zeros(N, H, D_h, device=device)
        o_edge.scatter_add_(
            0,
            src.view(-1, 1, 1).expand(-1, H, D_h),
            scale_e.unsqueeze(-1) * V_h[dst]
        )                                                                # (N, H, D_h)

        # ── Step 4: Two-pass Merge ────────────────────────────────────
        # Merge bucket and edge softmax statistics via log-sum-exp
        # Mathematically identical to softmax over all pairs P = P_LSH ∪ P_graph

        m_final = torch.maximum(m_bucket, m_edge)
        m_final = m_final.clamp(min=-1e9)   # guard: -inf - (-inf) = nan without this

        scale_b = torch.exp((m_bucket - m_final).clamp(min=-30))        # (N, H)
        scale_e = torch.exp((m_edge   - m_final).clamp(min=-30))        # (N, H)

        d_final = (d_bucket * scale_b + d_edge * scale_e).clamp(min=1e-6)
        o_final = (o_bucket * scale_b.unsqueeze(-1)
                 + o_edge   * scale_e.unsqueeze(-1))                    # (N, H, D_h)

        # ── Step 5: Normalise, dropout, project ───────────────────────
        h_global = o_final / d_final.unsqueeze(-1)                      # (N, H, D_h)
        h_global = F.dropout(h_global, p=self.dropout, training=self.training)
        h_global = h_global.view(N, self.hidden_dim)                    # (N, d)

        return self.out_proj(h_global), {"q": l_q, "k": l_k}