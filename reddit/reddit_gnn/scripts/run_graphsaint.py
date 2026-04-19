#!/usr/bin/env python3
"""
Run GraphSAINT baseline only — 3 seeds.
Usage:  python -m reddit_gnn.scripts.run_graphsaint
        python -m reddit_gnn.scripts.run_graphsaint --seeds 0

Note: Validation is done on CPU (full graph, val_mask applied).
      Training uses sampled random-walk subgraphs on GPU.
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE, SEEDS, NUM_CLASSES, NUM_FEATURES, DEFAULT_HPARAMS, set_seed
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.data.loaders import get_saint_loader
from reddit_gnn.models.graphsaint import GraphSAINTNet
from reddit_gnn.training.train_saint import train_saint
from reddit_gnn.training.utils import save_checkpoint, count_parameters
from reddit_gnn.evaluation.metrics import (
    get_test_predictions, compute_all_metrics, print_classification_report, aggregate_seeds
)
from reddit_gnn.evaluation.serialize import save_run_results


def main():
    parser = argparse.ArgumentParser(description="GraphSAINT Baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("GraphSAINT BASELINE")
    print(f"Device: {DEVICE}  |  Seeds: {args.seeds}")
    print(f"{'='*60}")

    data, _, _ = load_normalized_data()
    hp = DEFAULT_HPARAMS["graphsaint"]
    all_metrics = []

    for seed in args.seeds:
        set_seed(seed)

        model = GraphSAINTNet(
            in_channels=NUM_FEATURES, hidden_channels=hp["hidden"],
            out_channels=NUM_CLASSES, num_layers=hp["layers"],
            dropout=hp["dropout"],
        ).to(DEVICE)

        print(f"\n{'─'*60}")
        print(f"  GraphSAINT  seed={seed}  params={count_parameters(model):,}")
        print(f"{'─'*60}")

        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

        saint_loader = get_saint_loader(
            data, sampler_type=hp["sampler"], budget=hp["budget"],
            walk_length=hp["walk_length"], num_steps=hp["num_steps"],
            sample_coverage=hp["sample_coverage"],
        )

        # NOTE: train_saint validates on CPU using full-graph + val_mask
        # (not via NeighborLoader — SAINT batches lack seed-node-first ordering)
        history = train_saint(
            model, saint_loader, data, optimizer, DEVICE,
            max_epochs=hp["max_epochs"], patience=hp["patience"],
            model_name="GraphSAINT",
        )

        preds, labels = get_test_predictions(model, data, DEVICE, sparse_eval=True)
        metrics = compute_all_metrics(preds, labels, "graphsaint", f"baseline_seed{seed}")
        print_classification_report(preds, labels, "GraphSAINT")
        all_metrics.append(metrics)

        save_checkpoint(model, "graphsaint", seed=seed)
        save_run_results(metrics, history, model_name="graphsaint", seed=seed)
        print(f"  ✓ seed={seed}: acc={metrics['test_acc']:.4f}")

    if all_metrics:
        agg = aggregate_seeds(all_metrics)
        print(f"\n{'='*60}")
        print(f"  GraphSAINT AGGREGATE: acc={agg['test_acc_mean']:.4f} ± {agg['test_acc_std']:.4f}, "
              f"F1_macro={agg['f1_macro_mean']:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
