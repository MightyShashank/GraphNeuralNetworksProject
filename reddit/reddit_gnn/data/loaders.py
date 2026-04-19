"""
Stages 2C + 2D — Data Loaders for NeighborSampling and GraphSAINT.
Provides factories for training and inference loaders.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import SAINT_DIR


# ─── Stage 2C: NeighborLoader (GraphSAGE, GAT, GATv2) ──────────────────────


def get_train_loader(data, num_neighbors, batch_size, num_workers=0):
    """
    Training loader with bounded neighborhood sampling.

    Args:
        num_neighbors: list of ints, one per layer. e.g. [25, 10] for 2-layer.
        batch_size: Number of ROOT (seed) nodes per batch.
    """
    from torch_geometric.loader import NeighborLoader

    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,  # [25, 10] baseline
        batch_size=batch_size,         # 1024 root nodes
        input_nodes=data.train_mask,   # Sample roots from train set only
        shuffle=True,
        num_workers=num_workers,
    )


def get_inference_loader(data, batch_size=512, num_layers=2, num_workers=0):
    """
    Full-neighborhood inference loader — no sampling limit.
    Required for deterministic evaluation and embedding extraction.

    WARNING: Full neighborhood at 2 hops on Reddit (~492 avg degree)
    can be very memory-heavy. Use smaller batch_size (512 or 256).
    """
    from torch_geometric.loader import NeighborLoader

    return NeighborLoader(
        data,
        num_neighbors=[-1] * num_layers,  # -1 = full neighborhood
        batch_size=batch_size,
        input_nodes=None,                 # All nodes
        shuffle=False,                    # Preserve order for evaluation
        num_workers=num_workers,
    )


def get_val_loader(data, num_layers=2, batch_size=1024, num_workers=0, num_neighbors=None):
    """
    Validation/test loader — full neighborhood for accurate evaluation.
    Only processes val_mask nodes as roots.
    """
    from torch_geometric.loader import NeighborLoader

    if num_neighbors is None:
        num_neighbors = [-1] * num_layers

    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.val_mask,
        shuffle=False,
        num_workers=num_workers,
    )


def get_test_loader(data, num_layers=2, batch_size=1024, num_workers=0, num_neighbors=None):
    """
    Test loader — full neighborhood, test mask as roots.
    """
    from torch_geometric.loader import NeighborLoader

    if num_neighbors is None:
        num_neighbors = [-1] * num_layers

    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.test_mask,
        shuffle=False,
        num_workers=num_workers,
    )


# ─── Stage 2D: GraphSAINT Samplers ─────────────────────────────────────────


def get_saint_loader(
    data,
    sampler_type="rw",
    budget=6000,
    walk_length=2,
    num_steps=30,
    sample_coverage=100,
    save_dir=None,
    num_workers=0,
):
    """
    Create a GraphSAINT sampler.

    Args:
        sampler_type: 'rw' | 'node' | 'edge'
        budget: Approximate number of nodes per subgraph
        walk_length: Only for 'rw' sampler
        num_steps: Number of subgraph samples per epoch
        sample_coverage: Higher = more accurate normalization estimates
        save_dir: Cache directory for normalization coefficients
    """
    from torch_geometric.loader import (
        GraphSAINTRandomWalkSampler,
        GraphSAINTNodeSampler,
        GraphSAINTEdgeSampler,
    )

    if save_dir is None:
        save_dir = os.path.join(SAINT_DIR, sampler_type)
    os.makedirs(save_dir, exist_ok=True)

    common = dict(
        data=data,
        batch_size=budget,
        num_steps=num_steps,
        sample_coverage=sample_coverage,
        save_dir=save_dir,
        num_workers=num_workers,
    )

    if sampler_type == "rw":
        print(f"[2D] GraphSAINT RandomWalk sampler "
              f"(budget={budget}, walk_length={walk_length}, "
              f"num_steps={num_steps}, coverage={sample_coverage})")
        return GraphSAINTRandomWalkSampler(walk_length=walk_length, **common)
    elif sampler_type == "node":
        print(f"[2D] GraphSAINT Node sampler "
              f"(budget={budget}, num_steps={num_steps})")
        return GraphSAINTNodeSampler(**common)
    elif sampler_type == "edge":
        print(f"[2D] GraphSAINT Edge sampler "
              f"(budget={budget}, num_steps={num_steps})")
        return GraphSAINTEdgeSampler(**common)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")


# ─── ClusterLoader ──────────────────────────────────────────────────────────


def get_cluster_loader(cluster_data, clusters_per_batch=20, num_workers=0, shuffle=True):
    """Create a ClusterLoader from precomputed ClusterData."""
    from torch_geometric.loader import ClusterLoader

    return ClusterLoader(
        cluster_data,
        batch_size=clusters_per_batch,
        shuffle=shuffle,
        num_workers=num_workers,
    )
