"""
Stage 2B — ClusterGCN METIS Partitioning.
Partitions graph into clusters, caches results, analyzes partition quality.
"""

import shutil
import torch
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import CLUSTER_DIR


def prepare_cluster_gcn(data, num_parts, cache_dir=None):
    """
    Partition graph using METIS into num_parts clusters.
    PyG's ClusterData caches results automatically under save_dir.
    Re-running with same save_dir reloads cached partitions.
    """
    from torch_geometric.loader import ClusterData

    if cache_dir is None:
        cache_dir = os.path.join(CLUSTER_DIR, f"parts_{num_parts}")

    os.makedirs(cache_dir, exist_ok=True)

    print(f"  Partitioning into {num_parts} clusters (cache: {cache_dir})...")
    t0 = time.time()

    try:
        cluster_data = ClusterData(
            data,
            num_parts=num_parts,
            recursive=False,  # standard METIS (not recursive bisection)
            save_dir=cache_dir,
        )
    except RuntimeError as exc:
        # Corrupted cache (e.g. truncated zip from an interrupted previous run).
        # Wipe and re-partition from scratch.
        print(
            f"  WARNING: cache at {cache_dir} is corrupted ({exc}). "
            f"Deleting and re-partitioning..."
        )
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
        cluster_data = ClusterData(
            data,
            num_parts=num_parts,
            recursive=False,
            save_dir=cache_dir,
        )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    return cluster_data


def _get_perm_and_partptr(cluster_data):
    """
    Compatibility shim for PyG's ClusterData partition API.

    PyG >= 2.5: perm/partptr stored in cluster_data.partition namedtuple
                  as partition.node_perm / partition.partptr.
    PyG <  2.5: exposed directly as cluster_data.perm / cluster_data.partptr.
    """
    # New API (PyG >= 2.5 / 2.7): Partition namedtuple
    if hasattr(cluster_data, "partition"):
        return cluster_data.partition.node_perm, cluster_data.partition.partptr
    # Old API (PyG < 2.5): direct attributes
    if hasattr(cluster_data, "perm"):
        return cluster_data.perm, cluster_data.partptr
    raise AttributeError(
        "Cannot locate perm/partptr on ClusterData. "
        f"Available attrs: {[a for a in dir(cluster_data) if not a.startswith('__')]}"
    )


def analyze_partition_quality(cluster_data, data, num_parts):
    """
    Compute edge retention rate — fraction of edges falling WITHIN clusters.
    This is the key quality metric from Chiang et al. (2019).
    """
    perm, partptr = _get_perm_and_partptr(cluster_data)
    partition_id = torch.zeros(data.num_nodes, dtype=torch.long)

    for i in range(len(partptr) - 1):
        start = partptr[i]
        end = partptr[i + 1]
        partition_id[perm[start:end]] = i

    row, col = data.edge_index
    same_partition = partition_id[row] == partition_id[col]
    retention = same_partition.float().mean().item()

    # Compute cluster size statistics
    cluster_sizes = []
    for i in range(len(partptr) - 1):
        size = (partptr[i + 1] - partptr[i]).item()
        cluster_sizes.append(size)

    sizes_tensor = torch.tensor(cluster_sizes, dtype=torch.float)

    print(
        f"  num_parts={num_parts}: "
        f"edge_retention={retention:.4f} ({retention*100:.1f}%), "
        f"avg_cluster={sizes_tensor.mean():.0f}, "
        f"min_cluster={sizes_tensor.min():.0f}, "
        f"max_cluster={sizes_tensor.max():.0f}"
    )

    return retention, partition_id


def prepare_all_partitions(data, num_parts_list=None):
    """Run METIS for all partition configs needed for F1 ablation."""
    if num_parts_list is None:
        num_parts_list = [500, 1000, 1500, 3000, 6000]

    print("[2B] ClusterGCN METIS Partitioning:")
    results = {}

    for num_parts in num_parts_list:
        cache = os.path.join(CLUSTER_DIR, f"parts_{num_parts}")
        cluster_data = prepare_cluster_gcn(data, num_parts, cache)
        retention, partition_id = analyze_partition_quality(
            cluster_data, data, num_parts
        )
        results[num_parts] = {
            "cluster_data": cluster_data,
            "retention": retention,
            "partition_id": partition_id,
        }

    return results


def apply_diagonal_enhancement(batch, lambda_val):
    """
    Add lambda * I to the batch adjacency.
    Applied per mini-batch during training, not during preprocessing.
    """
    from torch_geometric.utils import add_self_loops

    edge_index, edge_weight = add_self_loops(
        batch.edge_index,
        edge_attr=batch.get("edge_weight", None),
        fill_value=lambda_val,
        num_nodes=batch.num_nodes,
    )
    batch.edge_index = edge_index
    batch.edge_weight = edge_weight
    return batch


if __name__ == "__main__":
    from reddit_gnn.data.normalize import load_normalized_data

    data, _, _ = load_normalized_data()
    prepare_all_partitions(data)
