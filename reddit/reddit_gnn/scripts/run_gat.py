#!/usr/bin/env python3
"""
Run GAT baseline only — 3 seeds.
Usage:  python -m reddit_gnn.scripts.run_gat
        python -m reddit_gnn.scripts.run_gat --seeds 0
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE, SEEDS, NUM_CLASSES, NUM_FEATURES, DEFAULT_HPARAMS, set_seed
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.data.loaders import get_train_loader, get_val_loader
from reddit_gnn.models.gat import GAT
from reddit_gnn.training.train_neighbor import train_neighbor_sampled
from reddit_gnn.training.utils import save_checkpoint, count_parameters
from reddit_gnn.evaluation.metrics import (
    get_test_predictions, compute_all_metrics, print_classification_report, aggregate_seeds
)
from reddit_gnn.evaluation.serialize import save_run_results


def main():
    parser = argparse.ArgumentParser(description="GAT Baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("GAT BASELINE")
    print(f"Device: {DEVICE}  |  Seeds: {args.seeds}")
    print(f"{'='*60}")

    data, _, _ = load_normalized_data()
    hp = DEFAULT_HPARAMS["gat"]
    all_metrics = []

    for seed in args.seeds:
        set_seed(seed)

        model = GAT(
            in_channels=NUM_FEATURES, out_channels=NUM_CLASSES,
            hidden_per_head=hp["hidden_per_head"], num_heads=hp["heads"],
            num_layers=hp["layers"], attn_dropout=hp["attn_dropout"],
            feat_dropout=hp["feat_dropout"],
        ).to(DEVICE)

        print(f"\n{'─'*60}")
        print(f"  GAT  seed={seed}  params={count_parameters(model):,}")
        print(f"{'─'*60}")

        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        train_loader = get_train_loader(data, hp["num_neighbors"], hp["batch_size"])
        val_loader = get_val_loader(data, num_layers=hp["layers"],
                                    num_neighbors=hp["num_neighbors"], batch_size=1024)

        history = train_neighbor_sampled(
            model, train_loader, val_loader, optimizer, DEVICE,
            max_epochs=hp["max_epochs"], patience=hp["patience"],
            model_name="GAT",
        )

        preds, labels = get_test_predictions(model, data, DEVICE)
        metrics = compute_all_metrics(preds, labels, "gat", f"baseline_seed{seed}")
        print_classification_report(preds, labels, "GAT")
        all_metrics.append(metrics)

        save_checkpoint(model, "gat", seed=seed)
        save_run_results(metrics, history, model_name="gat", seed=seed)
        print(f"  ✓ seed={seed}: acc={metrics['test_acc']:.4f}")

    if all_metrics:
        agg = aggregate_seeds(all_metrics)
        print(f"\n{'='*60}")
        print(f"  GAT AGGREGATE: acc={agg['test_acc_mean']:.4f} ± {agg['test_acc_std']:.4f}, "
              f"F1_macro={agg['f1_macro_mean']:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
