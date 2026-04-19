#!/usr/bin/env python3
"""
Preprocessing runner — execute all preprocessing steps (Phases 2 + 3).
Run: python scripts/run_preprocessing.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE


def main():
    print("=" * 60)
    print("REDDIT GNN — PREPROCESSING PIPELINE")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ── Phase 2: Global Preprocessing ──
    print("\n" + "=" * 60)
    print("PHASE 2: Global Preprocessing")
    print("=" * 60)

    # Step 1A: Download (just downloaind our reddit dataset)
    print("\n" + "=" * 60)
    print("STEP 1A: Downloading Reddit Dataset")
    from reddit_gnn.data.download import download_reddit
    data, dataset = download_reddit()
    print("=" * 60)

    # Step 1B: Normalize (Z-score normalization)
    print("\n" + "=" * 60)
    print("STEP 1B: Normalizing Reddit Dataset")
    from reddit_gnn.data.normalize import inspect_features, normalize_features
    inspect_features(data) # Just logging raw statistics
    data = normalize_features(data) 
    print("=" * 60)

    # Step 1C + 1D: Inspect (just checking our data)
    print("\n" + "=" * 60)
    print("STEP 1C + 1D: Inspecting Reddit Dataset")
    from reddit_gnn.data.inspect_graph import validate_masks, inspect_graph
    validate_masks(data)
    inspect_graph(data)

    # ── Phase 3: Model-Specific Preprocessing ──
    print("\n" + "=" * 60)
    print("PHASE 3: Model-Specific Preprocessing")
    print("=" * 60)

    # Step 2A: SGC feature precomputation
    from reddit_gnn.data.precompute_sgc import precompute_sgc_features
    precompute_sgc_features(data, max_K=5)

    # Step 2B: ClusterGCN METIS partitioning
    from reddit_gnn.data.partition_cluster import prepare_all_partitions
    prepare_all_partitions(data)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print("Next: Run baseline training with:")
    print("  python scripts/run_all_baselines.py")


if __name__ == "__main__":
    main()
