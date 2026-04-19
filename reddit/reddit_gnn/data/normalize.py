"""
Step 1B — Feature Inspection and Z-score Normalization.
Fits normalization on TRAIN nodes only to prevent data leakage.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import PREPROCESSED # We are going to save the normalised dataset here


def inspect_features(data) -> dict:
    """Log raw feature statistics before normalization."""
    x = data.x # Our node feature matrix
    stats = {
        "min": x.min().item(),
        "max": x.max().item(),
        "mean": x.mean().item(),
        "std": x.std().item(),
        "sparsity": (x == 0).float().mean().item(),
    }
    print("[1B] Raw feature statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    return stats


def normalize_features(data, save_path: str = None) -> "torch_geometric.data.Data":
    """
    Z-score normalization using TRAIN-only statistics.
    CRITICAL: Never fit on val/test — this would cause data leakage.
    """
    if save_path is None:
        save_path = os.path.join(PREPROCESSED, "reddit_normalized.pt")

    x = data.x  # [232965, 602]
    train_x = x[data.train_mask] # boolean indexing (we only want to train on the training nodes)

    # Fit statistics on training nodes only
    mean = train_x.mean(dim=0, keepdim=True)   # [1, 602]
    std = train_x.std(dim=0, keepdim=True)      # [1, 602]
    std[std == 0] = 1.0  # Avoid division by zero for zero-variance features

    # Transform ALL nodes using train statistics
    data.x = (x - mean) / std

    # Verify normalization on train split
    train_normed = data.x[data.train_mask]
    print(f"\n[1B] Post-normalization (train split):")
    print(f"  mean: {train_normed.mean().item():.6f} (should be ~0)")
    print(f"  std:  {train_normed.std().item():.6f} (should be ~1)")

    # Save normalized data + normalization parameters
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"data": data, "mean": mean, "std": std}, save_path)
    print(f"  Saved: {save_path}")

    return data


def load_normalized_data(path: str = None):
    """Load pre-normalized dataset from disk."""
    if path is None:
        path = os.path.join(PREPROCESSED, "reddit_normalized.pt")
    checkpoint = torch.load(path, weights_only=False)
    return checkpoint["data"], checkpoint["mean"], checkpoint["std"]


if __name__ == "__main__":
    from reddit_gnn.data.download import download_reddit

    data, _ = download_reddit()
    inspect_features(data)
    normalize_features(data)
