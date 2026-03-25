from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # ── Input dimensions ──────────────────────────────────────────────
    in_dim: int = 1433          # Cora=1433 | CiteSeer=3703 | PubMed=500 | arxiv=128
    hidden_dim: int = 128       # d — internal embedding dimension
    out_dim: int = 7            # Cora=7 | CiteSeer=6 | PubMed=3 | arxiv=40

    # ── Laplacian PE ──────────────────────────────────────────────────
    max_lap_k :int = 32
    threshold_on_lap_pe :float = 0.95
    lap_dim: int = 16           # k — number of eigenvectors

    # ── LSH Attention ─────────────────────────────────────────────────
    num_buckets: int = 8        # B — number of hash buckets
    num_heads: int = 4          # multi-head attention heads
    dropout: float = 0.5

    # ── Architecture ──────────────────────────────────────────────────
    num_layers: int = 4         # L — number of transformer layers
    ffn_dim: int = 256          # feed-forward hidden dim

    # ── Structural biases ─────────────────────────────────────────────
    max_spd: int = 5            # max shortest path distance to embed
    max_degree: int = 10        # max degree to embed
    num_spd_bins: int = 10

    # ── Recovery module ───────────────────────────────────────────────
    use_recovery: bool = True
    recovery_lambda: float = 0.25   # μ — weight of recovery loss

    # ── Hash supervision ──────────────────────────────────────────────
    use_hash_loss: bool = True
    hash_lambda: float = 0.3       # λ — weight of hash KL loss
    spd_gamma: float = 1.0         # γ in structural utility g()
    spd_delta: float = 0.5         # δ in structural utility g()

    # ── Training ──────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 50             # early stopping
    batch_size: int = 1            # full-batch for small graphs

    # ── Dataset ───────────────────────────────────────────────────────
    dataset_name: str = "Cora"     # Cora | CiteSeer | PubMed | ogbn-arxiv
    data_path: str = "./data"


# ── Quick dataset-specific overrides ─────────────────────────────────────────
DATASET_CONFIGS = {
    "Cora": dict(in_dim=1433, out_dim=7),
    "CiteSeer": dict(in_dim=3703, out_dim=6),
    "PubMed": dict(in_dim=500, out_dim=3),
    "ogbn-arxiv": dict(in_dim=128, out_dim=40, num_layers=6, hidden_dim=256),
}


def get_config(dataset_name: str = "Cora", **overrides) -> ModelConfig:
    cfg = ModelConfig(dataset_name=dataset_name)
    # Apply dataset-specific defaults
    for k, v in DATASET_CONFIGS.get(dataset_name, {}).items():
        setattr(cfg, k, v)
    # Apply any manual overrides
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg