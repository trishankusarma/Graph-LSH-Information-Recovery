# Training loop
"""
train.py
────────
Training loop for SparseGraphTransformer.

Run via:
    bash run_model.sh train Cora ./data
    python -m src.train --model-to-run Cora --data-path ./data
"""

import argparse
import time
import os
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from .data_loader              import load_dataset
from .model.transformer_model  import SparseGraphTransformer
from .hyperparameters.config   import get_config
from .losses.hash_loss         import hash_supervision_loss
from .losses.reconstruction_loss import recovery_loss


# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, data, adj, optimizer, config, device):
    model.train()
    optimizer.zero_grad()

    logits, aux = model(
        data.x.to(device),
        data.lap_pe.to(device),
        data.edge_index.to(device),
        data.deg.to(device),
    )

    y = data.y.to(device)

    # ── Task loss ────────────────────────────────────────────────────
    L_task = F.cross_entropy(logits[data.train_mask], y[data.train_mask])

    # ── Hash loss  (averaged over layers) ────────────────────────────
    L_hash = torch.tensor(0.0, device=device)
    if config.use_hash_loss:
        for bl in aux["bucket_logits"]:
            L_hash += hash_supervision_loss(
                bl, adj.to(device),
                data.lap_pe.to(device), config.num_buckets,
                delta=config.spd_delta,
            )
        L_hash /= len(aux["bucket_logits"])

    # ── Recovery loss  (last layer only to save memory) ──────────────
    L_rec = torch.tensor(0.0, device=device)
    if config.use_recovery and aux["confidences"][-1] is not None:
        L_rec = recovery_loss(
            h_recovered     = model.layers[-1].recovery.norm(
                              data.x.to(device)[:, :config.hidden_dim]   # placeholder
                              ) if False else aux["Vs"][-1],          # use V as proxy
            V               = aux["Vs"][-1],
            bucket_logits_q = aux["bucket_logits"][-1]["q"],
            confidence      = aux["confidences"][-1],
        )

    # ── Total loss ────────────────────────────────────────────────────
    loss = ((1 - config.hash_lambda) * L_task
            + config.hash_lambda     * L_hash
            + config.recovery_lambda * L_rec)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss"  : loss.item(),
        "L_task": L_task.item(),
        "L_hash": L_hash.item(),
        "L_rec" : L_rec.item(),
    }


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    logits, _ = model(
        data.x.to(device),
        data.lap_pe.to(device),
        data.edge_index.to(device),
        data.deg.to(device),
    )
    y    = data.y.to(device)
    pred = logits.argmax(dim=-1)

    results = {}
    for split, mask in [("train", data.train_mask),
                         ("val",   data.val_mask),
                         ("test",  data.test_mask)]:
        correct = (pred[mask] == y[mask]).sum().item()
        total   = mask.sum().item()
        results[split] = correct / total if total > 0 else 0.0
    return results


# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Config ───────────────────────────────────────────────────────
    config = get_config(dataset_name=args.model_to_run)
    config.data_path = args.data_path
    print(f"[Train] Config: {config}")

    # ── Data ─────────────────────────────────────────────────────────
    data, meta = load_dataset(
        config.dataset_name, config.data_path,
        max_lap_k = config.max_lap_k, threshold_on_lap_pe = config.threshold_on_lap_pe
    )

    # Dense adjacency for hash loss (small graphs only; use sparse for ogbn-arxiv)
    if meta["num_nodes"] <= 20_000:
        adj = to_dense_adj(data.edge_index, max_num_nodes=meta["num_nodes"]).squeeze(0)
    else:
        # Approximate with identity for very large graphs
        adj = torch.eye(meta["num_nodes"])

    # ── Model ─────────────────────────────────────────────────────────
    model = SparseGraphTransformer(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_test_acc  = 0.0
    patience_count = 0
    os.makedirs("best_models", exist_ok=True)

    print(f"\n{'Epoch':>6} {'Loss':>8} {'L_task':>8} {'L_hash':>8} "
          f"{'L_rec':>8} {'Train':>7} {'Val':>7} {'Test':>7} {'Time':>6}")
    print("─" * 72)

    for epoch in range(1, config.epochs + 1):
        t0     = time.time()
        losses = train_epoch(model, data, adj, optimizer, config, device)
        accs   = evaluate(model, data, device)
        scheduler.step()

        elapsed = time.time() - t0

        if accs["val"] > best_val_acc:
            best_val_acc  = accs["val"]
            best_test_acc = accs["test"]
            patience_count = 0
            torch.save(model.state_dict(),
                       f"best_models/{config.dataset_name}_best.pt")
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {losses['loss']:>8.4f} {losses['L_task']:>8.4f} "
                  f"{losses['L_hash']:>8.4f} {losses['L_rec']:>8.4f} "
                  f"{accs['train']:>7.4f} {accs['val']:>7.4f} "
                  f"{accs['test']:>7.4f} {elapsed:>5.1f}s")

        if patience_count >= config.patience:
            print(f"\n[Train] Early stopping at epoch {epoch}")
            break

    print(f"\n{'='*50}")
    print(f"Best Val Acc  : {best_val_acc:.4f}")
    print(f"Best Test Acc : {best_test_acc:.4f}")
    print(f"Model saved   : best_models/{config.dataset_name}_best.pt")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-to-run", type=str, default="Cora",
                        help="Dataset: Cora | CiteSeer | PubMed | ogbn-arxiv")
    parser.add_argument("--data-path",    type=str, default="./data")
    args = parser.parse_args()
    main(args)