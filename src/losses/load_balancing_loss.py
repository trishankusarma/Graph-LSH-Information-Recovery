import torch
import torch.nn.functional as F

def load_balancing_loss(bucket_logits):
    """
    Computes a loss that encourages uniform distribution of nodes across buckets.
    Based on the squared coefficient of variation (Standard in MoE).
    """
    # Convert logits to probabilities
    probs = F.softmax(bucket_logits, dim=-1)  # (N, B)
    
    # Get the mean probability of a node falling into each bucket
    mean_probs = probs.mean(dim=0)            # (B,)
    
    # Ideal mean_probs is 1/B for all buckets. 
    # This loss minimizes the variance of the bucket assignments.
    num_buckets = probs.size(-1)
    loss = num_buckets * torch.sum(mean_probs ** 2) - 1.0
    
    return loss