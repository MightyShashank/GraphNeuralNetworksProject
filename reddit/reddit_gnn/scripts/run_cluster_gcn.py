#!/usr/bin/env python3
"""
Run ClusterGCN baseline only — 3 seeds.
Usage:  python -m reddit_gnn.scripts.run_cluster_gcn
        python -m reddit_gnn.scripts.run_cluster_gcn --seeds 0

Note: Validation is done on CPU (full graph, val_mask applied).
      Training uses METIS-partitioned subgraphs on GPU.
      ClusterLoader uses num_workers=0 to prevent CUDA fork crash.
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE, SEEDS, NUM_CLASSES, NUM_FEATURES, DEFAULT_HPARAMS, set_seed
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.data.partition_cluster import prepare_cluster_gcn
from reddit_gnn.models.cluster_gcn import ClusterGCN
from reddit_gnn.training.train_cluster import train_cluster_gcn
from reddit_gnn.training.utils import save_checkpoint, count_parameters
from reddit_gnn.evaluation.metrics import (
    get_test_predictions, compute_all_metrics, print_classification_report, aggregate_seeds
)
from reddit_gnn.evaluation.serialize import save_run_results


def main():
    parser = argparse.ArgumentParser(description="ClusterGCN Baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("ClusterGCN BASELINE")
    print(f"Device: {DEVICE}  |  Seeds: {args.seeds}")
    print(f"{'='*60}")

    data, _, _ = load_normalized_data()
    hp = DEFAULT_HPARAMS["cluster_gcn"]

    # Partition once — reused across seeds
    print(f"\n  Preparing METIS partitions (num_parts={hp['num_parts']})...")
    cluster_data = prepare_cluster_gcn(data, hp["num_parts"])

    all_metrics = []

    for seed in args.seeds:
        set_seed(seed)

        model = ClusterGCN(
            in_channels=NUM_FEATURES, hidden_channels=hp["hidden"],
            out_channels=NUM_CLASSES, num_layers=hp["layers"],
            dropout=hp["dropout"],
        ).to(DEVICE)

        print(f"\n{'─'*60}")
        print(f"  ClusterGCN  seed={seed}  params={count_parameters(model):,}")
        print(f"{'─'*60}")

        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

        # NOTE: train_cluster_gcn validates on CPU using full-graph + val_mask
        # (ClusterLoader batches lack seed-node-first ordering NeighborLoader provides)
        history = train_cluster_gcn(
            model, cluster_data, data, optimizer, DEVICE,
            clusters_per_batch=hp["clusters_per_batch"],
            lambda_val=hp["lambda_val"],
            max_epochs=hp["max_epochs"], patience=hp["patience"],
            model_name="ClusterGCN",
        )

        preds, labels = get_test_predictions(model, data, DEVICE, sparse_eval=True)
        metrics = compute_all_metrics(preds, labels, "cluster_gcn", f"baseline_seed{seed}")
        print_classification_report(preds, labels, "ClusterGCN")
        all_metrics.append(metrics)

        save_checkpoint(model, "cluster_gcn", seed=seed)
        save_run_results(metrics, history, model_name="cluster_gcn", seed=seed)
        print(f"  ✓ seed={seed}: acc={metrics['test_acc']:.4f}")

    if all_metrics:
        agg = aggregate_seeds(all_metrics)
        print(f"\n{'='*60}")
        print(f"  ClusterGCN AGGREGATE: acc={agg['test_acc_mean']:.4f} ± {agg['test_acc_std']:.4f}, "
              f"F1_macro={agg['f1_macro_mean']:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
