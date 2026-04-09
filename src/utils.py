from datetime import datetime
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

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

def plot_bucket_distribution(bucket_logits, num_buckets, epoch, dataset_name, save_dir="plots"):
    """
    Plots the distribution of nodes across LSH buckets.
    
    Args:
        bucket_logits : dict containing 'q' and 'k' logits from the model
        num_buckets   : int, the number of buckets (B) configured in the model
        epoch         : int, current epoch (for the title/filename)
        dataset_name  : str, e.g., 'Cora'
        save_dir      : str, directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Get hard bucket assignments
    # Detach from graph and move to CPU for plotting
    bq = bucket_logits['q'].argmax(dim=-1).detach().cpu()
    bk = bucket_logits['k'].argmax(dim=-1).detach().cpu()
    
    # 2. Count the number of nodes in each bucket
    q_counts = torch.bincount(bq, minlength=num_buckets).numpy()
    k_counts = torch.bincount(bk, minlength=num_buckets).numpy()
    
    # 3. Plotting Setup
    x = np.arange(num_buckets)
    width = 0.35  # width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Side-by-side bars for Queries and Keys
    rects1 = ax.bar(x - width/2, q_counts, width, label='Queries', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, k_counts, width, label='Keys', color='salmon', edgecolor='black')
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Bucket ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_title(f'LSH Bucket Distribution - {dataset_name} (Epoch {epoch})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'B{i}' for i in range(num_buckets)])
    ax.legend()
    
    # Add a dashed line showing the "Ideal/Perfectly Balanced" distribution
    total_nodes = len(bq)
    ideal_count = total_nodes / num_buckets
    ax.axhline(ideal_count, color='gray', linestyle='--', label=f'Ideal uniform ({ideal_count:.1f}/bucket)')
    ax.legend()
    
    # Optional: Attach a text label above each bar displaying its height
    ax.bar_label(rects1, padding=3, fontsize=9)
    ax.bar_label(rects2, padding=3, fontsize=9)
    
    fig.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, f"{dataset_name}_bucket_dist_ep{epoch}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()