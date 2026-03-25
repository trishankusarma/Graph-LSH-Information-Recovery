# Node classification metrics
"""
evaluate.py
───────────
Evaluation script — loads best saved model and reports metrics.

Run via:
    bash run_model.sh eval Cora ./data
    python -m src.evaluate --model-to-run Cora --data-path ./data
"""

import argparse
import torch
from sklearn.metrics import classification_report

from .data_loader             import load_dataset
from .model.transformer_model import SparseGraphTransformer
from .hyperparameters.config  import get_config
from .utils import log

@torch.no_grad()
def full_evaluate(model, data, device, meta):
    model.eval()
    logits, aux = model(
        data.x.to(device),
        data.lap_pe.to(device),
        data.edge_index.to(device),
        data.spd.to(device),
        data.deg.to(device),
    )
    y    = data.y.to(device)
    pred = logits.argmax(dim=-1)

    log("\n" + "="*50)
    for split, mask in [("Train", data.train_mask),
                         ("Val",   data.val_mask),
                         ("Test",  data.test_mask)]:
        correct = (pred[mask] == y[mask]).sum().item()
        total   = mask.sum().item()
        acc     = correct / total if total > 0 else 0.0
        log(f"{split:>6} Accuracy : {acc:.4f}  ({correct}/{total})")

    # Detailed test report
    log("\n── Test Set Classification Report ──")
    log(classification_report(
        y[data.test_mask].cpu().numpy(),
        pred[data.test_mask].cpu().numpy(),
        digits=4,
    ))

    # Confidence analysis
    if aux["confidences"][-1] is not None:
        conf = aux["confidences"][-1]
        log(f"\n── Recovery Module Confidence ──")
        log(f"  Mean confidence  : {conf.mean():.4f}")
        log(f"  Std  confidence  : {conf.std():.4f}")
        log(f"  Low-conf nodes   : {(conf < 0.3).sum().item()} "
              f"({(conf < 0.3).float().mean()*100:.1f}%)")
        log(f"  High-conf nodes  : {(conf > 0.7).sum().item()} "
              f"({(conf > 0.7).float().mean()*100:.1f}%)")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = get_config(dataset_name=args.model_to_run)
    cfg.data_path = args.data_path

    data, meta = load_dataset(
        cfg.dataset_name, cfg.data_path,
        lap_dim=cfg.lap_dim, max_spd=cfg.max_spd
    )

    model = SparseGraphTransformer(cfg).to(device)
    ckpt  = f"best_models/{cfg.dataset_name}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    log(f"[Eval] Loaded checkpoint: {ckpt}")

    full_evaluate(model, data, device, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-to-run", type=str, default="Cora")
    parser.add_argument("--data-path",    type=str, default="./data")
    args = parser.parse_args()
    main(args)