from datetime import datetime
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import os

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Plotting
def plot_training(history: dict, dataset_name: str, save_dir: str):
    """
    Plots two figures:
      1. Loss curves  : total loss, L_task, L_hash, L_rec vs epoch
      2. Accuracy     : train, val, test accuracy vs epoch

    Saves to save_dir/{dataset_name}_losses.png and _accuracy.png
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = history["epochs"]

    # ── Figure 1 — Loss curves ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{dataset_name} — Training Curves", fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(epochs, history["loss"],   label="Total Loss",   color="#2c7bb6", linewidth=2)
    ax.plot(epochs, history["L_task"], label="Task Loss",    color="#d7191c", linewidth=1.5, linestyle="--")
    ax.plot(epochs, history["L_hash"], label="Hash Loss",    color="#1a9641", linewidth=1.5, linestyle="--")
    ax.plot(epochs, history["L_rec"],  label="Recovery Loss",color="#f07d02", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")   # log scale — recovery loss is much smaller than task loss

    # ── Figure 2 — Accuracy curves ────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train",
            color="#2c7bb6", linewidth=2)
    ax.plot(epochs, history["val_acc"],   label="Val",
            color="#d7191c", linewidth=2)
    ax.plot(epochs, history["test_acc"],  label="Test",
            color="#1a9641", linewidth=2)

    # Mark best val epoch
    best_val_epoch = epochs[history["val_acc"].index(max(history["val_acc"]))]
    best_val_acc   = max(history["val_acc"])
    ax.axvline(x=best_val_epoch, color="gray", linestyle=":", alpha=0.7)
    ax.annotate(
        f"Best Val\n{best_val_acc:.4f}",
        xy=(best_val_epoch, best_val_acc),
        xytext=(best_val_epoch + max(epochs) * 0.03, best_val_acc - 0.05),
        fontsize=8,
        color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{dataset_name}_training.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"[Plot] Saved → {path}")

    # ── Figure 3 — Recovery loss alone (linear scale for detail) ─────
    if any(v > 0 for v in history["L_rec"]):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history["L_rec"], color="#f07d02", linewidth=2, label="Recovery Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L_rec")
        ax.set_title(f"{dataset_name} — Recovery Loss (linear)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path_rec = os.path.join(save_dir, f"{dataset_name}_recovery_loss.png")
        plt.savefig(path_rec, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"[Plot] Saved → {path_rec}")
