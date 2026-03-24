"""
data_loader.py
──────────────
Unified loader for:
  • Planetoid  : Cora, CiteSeer, PubMed
  • OGB        : ogbn-arxiv

Adds:
  • Laplacian Positional Encodings  (λ ∈ R^{N×k})
  • Node degree vector              (deg ∈ R^N)

Usage:
    data, meta = load_dataset("Cora",      data_path="./data")
    data, meta = load_dataset("ogbn-arxiv", data_path="./data")
"""

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import (
    to_scipy_sparse_matrix,
    get_laplacian,
    degree,
    to_undirected,
)
import time
from .utils import log

try:
    from ogb.nodeproppred import PygNodePropPredDataset
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False

# HARDCODED_PATHS
DATASET_PATH = "./data" 

# Laplacian PE
def compute_laplacian_pe(edge_index, num_nodes: int, max_k: int = 64, threshold_on_lap_pe :float = 0.95) -> torch.Tensor:
    """
    Compute the top-k non-trivial eigenvectors of the normalised graph Laplacian.
    Returns λ ∈ R^{N×k}.
    """
    # Normalised Laplacian via PyG helper
    lap_idx, lap_val = get_laplacian(
        edge_index, normalization="sym", num_nodes=num_nodes
    )
    L = to_scipy_sparse_matrix(lap_idx, lap_val, num_nodes).astype(np.float32)

    # k+1 smallest eigenvalues; skip trivial λ=0
    try:
        eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=max_k + 1, which="SM")

    except Exception as e:
        # Fallback: dense eigen for tiny graphs
        log(f"Fallback error : {e}")
        L_dense = L.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)

    log(f"Eigen values found : {len(eigenvalues)}")
    # Top k eigen values 
    all_eigen_values_sum = sum(eigenvalues[1:])
    cumm_eigen_values = 0
    eigen_values_to_consider_left, eigen_values_to_consider_right = -1, len(eigenvalues) - 1

    for index, eigenvalue in enumerate(eigenvalues[1:], start=1):
        if eigenvalue > 1e-6 and eigen_values_to_consider_left == -1:
            log("Eigen value set for left index")
            eigen_values_to_consider_left = index

        if eigen_values_to_consider_left == -1:
            continue

        cumm_eigen_values = cumm_eigen_values + eigenvalue/all_eigen_values_sum

        if cumm_eigen_values >= threshold_on_lap_pe:
            eigen_values_to_consider_right = index
            break
    
    eigen_values_to_consider_right = min(eigen_values_to_consider_right, eigen_values_to_consider_left + max_k - 1)
        
    eigenvectors = eigenvectors[:, eigen_values_to_consider_left:eigen_values_to_consider_right+1]
    log(f"Dynamic k selected: ({eigen_values_to_consider_left} :: {eigen_values_to_consider_right}) / {len(eigenvalues)}")
    pe = torch.from_numpy(eigenvectors).float()      # (N, k)
    return pe                                        # (N, k)

# Main loader
def load_dataset(dataset_name: str, data_path: str = DATASET_PATH,
                 max_lap_k: int = 64, threshold_on_lap_pe: float = 0.95):
    """
    Returns
    -------
    data : torch_geometric.data.Data  — with extra attributes:
               data.lap_pe  (N, max_lap_k)
               data.spd     (N, N)      long tensor
               data.deg     (N,)        long tensor
    meta : dict  — num_nodes, num_classes, num_features, split info
    """
    log(f"[DataLoader] Loading {dataset_name} ...")
    start_time = time.time()

    # Load raw dataset
    if dataset_name in ("Cora", "CiteSeer", "PubMed"):
        dataset = Planetoid(
            root=f"{data_path}/Planetoid",
            name=dataset_name,
            transform=NormalizeFeatures(),
        )
        data = dataset[0]
        split = "planetoid"                          # uses data.train_mask etc.

    elif dataset_name == "ogbn-arxiv":
        if not OGB_AVAILABLE:
            raise ImportError("ogb not installed. Run: pip install ogb")
        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv", root=f"{data_path}/OGB"
        )
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)   # make undirected
        split_idx = dataset.get_idx_split()
        data.train_mask = _idx_to_mask(split_idx["train"], data.num_nodes)
        data.val_mask   = _idx_to_mask(split_idx["valid"], data.num_nodes)
        data.test_mask  = _idx_to_mask(split_idx["test"],  data.num_nodes)
        data.y = data.y.squeeze(1)
        split = "ogb"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from: Cora, CiteSeer, PubMed, ogbn-arxiv")

    num_nodes = data.num_nodes
    num_classes = dataset.num_classes
    log(f"Number of nodes: {num_nodes}")
    log(f"Number of edges: {data.num_edges}")
    log(f"Node feature shape: {data.x.shape}")
    log(f"Labels shape: {data.y.shape}")
    log(f"Edge_index shape {data.edge_index.shape}")
    log(f"Number of classes {num_classes}")

    # ── Laplacian PE ─────────────────────────────────────────────────
    log(f"[DataLoader] Computing Laplacian PE (k={max_lap_k}) ...")
    data.lap_pe = compute_laplacian_pe(data.edge_index, num_nodes, max_k=max_lap_k, threshold_on_lap_pe=threshold_on_lap_pe)

    # ── Degree ───────────────────────────────────────────────────────
    data.deg = degree(data.edge_index[0], num_nodes=num_nodes).long()

    meta = {
        "dataset_name": dataset_name,
        "num_nodes": num_nodes,
        "num_features": data.num_features,
        "num_classes": num_classes,
        "split": split,
        "lap_dim": data.lap_pe.shape[1],
    }
    log(f"[DataLoader] Done. {meta}")
    log(f"Time taken to load for {dataset_name}: {time.time() - start_time}")
    return data, meta

# Helpers
def _idx_to_mask(idx: torch.Tensor, num_nodes: int) -> torch.BoolTensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask

if __name__ == "__main__":
    from .hyperparameters.config import Config
    config = Config()
    for name in ["Cora", "CiteSeer", "PubMed", "ogbn-arxiv"]:
        d, m = load_dataset(name, data_path="./data", max_lap_k = config.max_lap_k, threshold_on_lap_pe = config.threshold_on_lap_pe)
        log(f"{name}: x={d.x.shape} | lap_pe={d.lap_pe.shape} | "
              f"deg={d.deg.shape}")
        log(f"  train={d.train_mask.sum()} | val={d.val_mask.sum()} | "
              f"test={d.test_mask.sum()}\n")