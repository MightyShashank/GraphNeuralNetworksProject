"""
Steps 1C + 1D — Split Handling, Mask Validation, and Graph Connectivity Inspection.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import PREPROCESSED


# Checks whether every node belongs to exactly one split. Its a disgnostic check.
# For each split it counts how many nodes belong to each class, prints min, max class counts and also prints imbalance ratio.

def validate_masks(data):
    """
    Step 1C — Confirm masks are non-overlapping and cover all nodes.
    Check class distribution across splits.
    """
    print("[1C] Mask Validation:")

    # Non-overlapping checks
    assert (data.train_mask & data.val_mask).sum() == 0, "Train/Val overlap!"
    assert (data.train_mask & data.test_mask).sum() == 0, "Train/Test overlap!"
    assert (data.val_mask & data.test_mask).sum() == 0, "Val/Test overlap!"
    print("  ✓ No mask overlaps")

    # Coverage check
    total = data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()
    print(f"  All nodes accounted for: {total.item() == data.num_nodes}")

    # Class distribution per split
    print("\n  Class distribution per split:")
    for split, mask in [("Train", data.train_mask),
                        ("Val", data.val_mask),
                        ("Test", data.test_mask)]:
        labels = data.y[mask]
        counts = labels.bincount(minlength=41)
        min_c = counts.min().item()
        max_c = counts.max().item()
        ratio = max_c / max(min_c, 1)
        print(f"    {split:5s}: min_class={min_c}, max_class={max_c}, ratio={ratio:.1f}x")


def inspect_graph(data, save_dir: str = None):
    """
    Step 1D — Graph connectivity inspection.
    Computes degree stats, checks structural properties,
    and computes per-node homophily.
    """
    from torch_geometric.utils import (
        is_undirected,
        contains_self_loops,
        contains_isolated_nodes,
        degree,
    )

    if save_dir is None:
        save_dir = PREPROCESSED

    print("\n[1D] Graph Connectivity Inspection:")

    # Structural checks
    undirected = is_undirected(data.edge_index)
    self_loops = contains_self_loops(data.edge_index)
    isolated = contains_isolated_nodes(data.edge_index, data.num_nodes)

    print(f"  Is undirected:       {undirected}")
    print(f"  Contains self-loops: {self_loops}")
    print(f"  Isolated nodes:      {isolated}")

    if not undirected:
        print("  ⚠️  Graph is directed! Consider converting with to_undirected().")
    if self_loops:
        print("  ⚠️  Self-loops found! Remove before adding model-specific self-loops.")

    # Degree statistics
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    print(f"\n  Degree statistics:")
    print(f"    Mean degree:  {deg.mean().item():.1f}")
    print(f"    Max degree:   {deg.max().item():.0f}")
    print(f"    Min degree:   {deg.min().item():.0f}")
    print(f"    Std degree:   {deg.std().item():.1f}")
    print(f"    Median degree: {deg.median().item():.0f}")

    # Degree distribution buckets
    for lo, hi, name in [(0, 10, "1-10"), (10, 50, "11-50"),
                         (50, 200, "51-200"), (200, 1000, "201-1000"),
                         (1000, float("inf"), ">1000")]:
        count = ((deg >= lo) & (deg < hi)).sum().item()
        print(f"    Degree {name}: {count:,} nodes ({count/data.num_nodes*100:.1f}%)")

    # Per-node homophily
    print(f"\n  Computing per-node homophily...")
    h_v = _compute_node_homophily(data)
    save_path = os.path.join(save_dir, "node_homophily.pt")
    torch.save(h_v, save_path)
    print(f"  Global homophily (mean h_v): {h_v.mean().item():.4f}")
    print(f"  h_v std:  {h_v.std().item():.4f}")
    print(f"  h_v min:  {h_v.min().item():.4f}")
    print(f"  h_v max:  {h_v.max().item():.4f}")

    # Homophily regime distribution
    regime1 = (h_v < 0.3).sum().item()
    regime_mid = ((h_v >= 0.3) & (h_v < 0.7)).sum().item()
    regime2 = (h_v >= 0.7).sum().item()
    print(f"\n  Homophily regime distribution:")
    print(f"    Regime 1 (h_v < 0.3):   {regime1:,} nodes ({regime1/data.num_nodes*100:.1f}%)")
    print(f"    Mid (0.3 <= h_v < 0.7): {regime_mid:,} nodes ({regime_mid/data.num_nodes*100:.1f}%)")
    print(f"    Regime 2/3 (h_v >= 0.7): {regime2:,} nodes ({regime2/data.num_nodes*100:.1f}%)")

    print(f"\n  Saved: {save_path}")
    return deg, h_v


def _compute_node_homophily(data) -> torch.Tensor:
    """
    Compute per-node homophily: fraction of neighbors with the same label.
    h_v = |{u in N(v) : y_u == y_v}| / |N(v)|
    """
    row, col = data.edge_index
    same_label = (data.y[row] == data.y[col]).float()

    from torch_geometric.utils import degree
    deg = degree(row, num_nodes=data.num_nodes)

    # Sum same-label edges per node
    h_v = torch.zeros(data.num_nodes)
    h_v.scatter_add_(0, row, same_label)

    # Normalize by degree
    mask = deg > 0
    h_v[mask] = h_v[mask] / deg[mask]
    h_v[~mask] = 0.0  # Isolated nodes get 0 homophily

    return h_v


if __name__ == "__main__":
    from reddit_gnn.data.normalize import load_normalized_data

    data, _, _ = load_normalized_data()
    validate_masks(data)
    inspect_graph(data)
