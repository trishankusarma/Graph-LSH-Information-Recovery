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
        N, H, D_h = Q.size(0), self.num_heads, self.head_dim
        device = Q.device

        # Get the max allowed index for your embedding table :: to be needed later
        max_deg_val = self.deg_src_emb.num_embeddings - 1

        # ── Step 1: Bucket logits ─────────────────────────────────────
        q_input = torch.cat([Q, lap_pe], dim=-1)    # (N, d+k)
        k_input = torch.cat([K, lap_pe], dim=-1)

        l_q = self.mlp_q(q_input)                   # (N, B)
        l_k = self.mlp_k(k_input)                   # (N, B)

        bq = l_q.argmax(dim=-1)                     # (N,)  — query bucket ids
        bk = l_k.argmax(dim=-1)                     # (N,)  — key   bucket ids

        # Step 2 : Sort nodes by bucket ID
        # We sort Queries and Keys separately because they might land in different buckets
        q_idx = torch.argsort(bq)
        k_idx = torch.argsort(bk)

        # Step 3 : Reorder everything
        Q_s, K_s, V_s = Q[q_idx], K[k_idx], V[k_idx]
        PE_q_s, PE_k_s = lap_pe[q_idx], lap_pe[k_idx]
        deg_q_s, deg_k_s = deg[q_idx], deg[k_idx]

        # Step 4 : Create Nested Tensors
        # Find bucket sizes (e.g., how many Queries in bucket 0, 1, 2...)
        q_counts = torch.bincount(bq, minlength=self.num_buckets)
        k_counts = torch.bincount(bk, minlength=self.num_buckets)

        # Step 4.1 : Calculate offsets (starting positions) for each bucket
        # [0, count1, count1+count2, ...]
        q_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=Q.device), q_counts.cumsum(0)])
        k_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=Q.device), k_counts.cumsum(0)])

        q_list = []
        k_list = []
        v_list = []
        bias_list = []

        # Step 5 :: iterate over buckets and find the attention scores
        for b in range(self.num_buckets):
            # Step 5.1 : Get the start and end indices for this bucket in the sorted arrays
            s_q, e_q = q_offsets[b], q_offsets[b+1]
            s_k, e_k = k_offsets[b], k_offsets[b+1]

            # Skip if either side is empty (no attention possible)
            if s_q == e_q or s_k == e_k:
                continue

            # Step 5.2 : Slice the sorted tensors
            q_b = Q_s[s_q:e_q].view(-1, H, D_h).transpose(0, 1) # (H, Nq, Dh)
            k_b = K_s[s_k:e_k].view(-1, H, D_h).transpose(0, 1) # (H, Nk, Dh)
            v_b = V_s[s_k:e_k].view(-1, H, D_h).transpose(0, 1) # (H, Nk, Dh)

            # Step 5.3 : Efficient PE Distance (Difference) inside the bucket
            # shape: (Nq, Nk)
            pe_q_b = PE_q_s[s_q:e_q]
            pe_k_s_b = PE_k_s[s_k:e_k]

            # Step 5.4 : Square distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab^T
            dist_sq = (pe_q_b**2).sum(-1, keepdim=True) + \
                    (pe_k_s_b**2).sum(-1, keepdim=True).T - \
                    2 * torch.matmul(pe_q_b, pe_k_s_b.T)

            # Step 5.5 : Take sqrt and bucketize for the SPD bias
            dist = torch.sqrt(dist_sq.clamp(min=1e-9))
            spd_idx = torch.bucketize(dist, self.spd_boundaries)

            # Step 5.6 : Construct the Bias (H, Nq, Nk)
            # SPD bias lookup
            b_spd = self.spd_bias(spd_idx) # (Nq, Nk, H)

            # Step 5.7 : Degree bias lookup
            # Clamp the degrees so they never exceed that max index
            d_q_clamped = deg_q_s[s_q:e_q].clamp(0, max_deg_val)
            d_k_clamped = deg_k_s[s_k:e_k].clamp(0, max_deg_val)

            # 3. Now perform the lookup safely
            b_deg = self.deg_src_emb(d_q_clamped).unsqueeze(1) + \
                    self.deg_dst_emb(d_k_clamped).unsqueeze(0) # (Nq, Nk, H)

            # Combine and permute to (H, Nq, Nk) to match SDPA requirements
            bias_b = (b_spd + b_deg).permute(2, 0, 1)

            q_list.append(q_b)
            k_list.append(k_b)
            v_list.append(v_b)
            bias_list.append(bias_b)

        # Step 6 : 
        out_list = []
        for i in range(len(q_list)):
            # q_b: (H, Nq, Dh), k_b: (H, Nk, Dh), bias_b: (H, Nq, Nk)
            
            # We call SDPA on regular tensors one bucket at a time
            # This allows us to use the structural bias as the mask
            bucket_out = F.scaled_dot_product_attention(
                q_list[i], 
                k_list[i], 
                v_list[i], 
                attn_mask=bias_list[i],
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
            out_list.append(bucket_out)
        
        # ── Step 7: Restoration ───────────────────────
        # 7.1 Create a buffer for the reordered output
        # Shape: (N, H, head_dim)
        h_lsh = torch.zeros(N, self.num_heads, self.head_dim, device=device)

        # 7.2 Use the offsets to scatter the results back
        out_ptr = 0  # Pointer to track which element of out_list we are using
        for i, b in enumerate(range(self.num_buckets)):
            s_q, e_q = q_offsets[b], q_offsets[b+1]
            if s_q == e_q or k_offsets[b] == k_offsets[b+1]:
                continue
            
            # The i-th element in out_list corresponds to bucket b
            # We move (H, Nq, Dh) -> (Nq, H, Dh)
            # Get the output for THIS specific non-empty bucket
            bucket_output = out_list[out_ptr].transpose(0, 1) # (Nq_b, H, Dh)
            
            # Map back to original indices using the sorted index map
            h_lsh[q_idx[s_q:e_q]] = bucket_output
            
            out_ptr += 1 # Move to the next result in out_list

        # Step 8 : Process Graph Edges: Handle the edge_index pairs that LSH might have missed
        # 8.1 Get source and target nodes for graph edges
        src, dst = edge_index[0], edge_index[1]

        # 8.2 Compute scores for these edges
        # Reshape Q, K, V to (N, H, Dh) first
        Q_h = Q.view(N, self.num_heads, self.head_dim)
        K_h = K.view(N, self.num_heads, self.head_dim)
        V_h = V.view(N, self.num_heads, self.head_dim)

        q_e = Q_h[src] # (E, H, Dh)
        k_e = K_h[dst] # (E, H, Dh)

        # Dot product over the head_dim: (E, H)
        edge_scores = (q_e * k_e).sum(-1) / (self.head_dim ** 0.5)

        # 8.3 Apply the same structural biases to the edges
        # PE distance bias
        # Ensure the biases are (E, H)
        edge_pe_dist = (lap_pe[src] - lap_pe[dst]).norm(dim=-1)
        edge_spd_bin = torch.bucketize(edge_pe_dist, self.spd_boundaries)
        edge_scores = edge_scores + self.spd_bias(edge_spd_bin)

        # Degree bias
        edge_deg_src = deg[src].clamp(0, max_deg_val)
        edge_deg_dst = deg[dst].clamp(0, max_deg_val)
        edge_scores = edge_scores + self.deg_src_emb(edge_deg_src) + self.deg_dst_emb(edge_deg_dst)

        # 8.4 Sparse Softmax and Aggregation
        edge_attn = pyg_softmax(edge_scores, src, num_nodes=N)
        edge_attn = self.attn_drop(edge_attn)
        
        # Aggregate values: Σ α_ij * V_j
        h_graph = torch.zeros(N, self.num_heads, self.head_dim, device=device)
        # Reshape src for scattering: (E) -> (E, 1, 1) then expand to (E, H, Dh)
        index = src.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim)
        values = edge_attn.unsqueeze(-1) * V_h[dst] # (E, H, 1) * (E, H, Dh) -> (E, H, Dh)

        h_graph.scatter_add_(0, index, values)

        # 9.1 Combine branches (Sum or Gated Sum) :: Final Combination and Projection
        # For now, a simple weighted sum or residual-style addition
        h_total = h_lsh + h_graph

        # 9.2 Final linear projection
        h_total = h_total.view(N, self.hidden_dim)
        out = self.out_proj(h_total)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out, {"q": l_q, "k": l_k}