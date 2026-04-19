#!/usr/bin/env python3
"""
Run SGC baseline only — 3 seeds.
Usage:  python -m reddit_gnn.scripts.run_sgc
        python -m reddit_gnn.scripts.run_sgc --seeds 0
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE, SEEDS, SGC_DIR, DEFAULT_HPARAMS, set_seed
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.training.train_sgc import train_sgc
from reddit_gnn.evaluation.metrics import compute_all_metrics, print_classification_report, aggregate_seeds
from reddit_gnn.training.utils import save_checkpoint
from reddit_gnn.evaluation.serialize import save_run_results


def main():
    parser = argparse.ArgumentParser(description="SGC Baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("SGC BASELINE")
    print(f"Device: {DEVICE}  |  Seeds: {args.seeds}")
    print(f"{'='*60}")

    data, _, _ = load_normalized_data()
    hp = DEFAULT_HPARAMS["sgc"]
    all_metrics = []

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n{'─'*60}")
        print(f"  SGC  seed={seed}")
        print(f"{'─'*60}")

        model, history = train_sgc(
            K=hp["K"], data=data, device=DEVICE,
            max_epochs=hp["max_epochs"], lr=hp["lr"],
            weight_decay=hp["weight_decay"], patience=hp["patience"],
        )

        # Evaluate on precomputed features
        X_K = torch.load(os.path.join(SGC_DIR, f"reddit_sgc_K{hp['K']}.pt"),
                         weights_only=False).to(DEVICE)
        model.eval()
        with torch.no_grad():
            out = model(X_K[data.test_mask.to(DEVICE)])
            preds = out.argmax(1).cpu().numpy()
            labels = data.y[data.test_mask].numpy()

        metrics = compute_all_metrics(preds, labels, "sgc", f"baseline_K{hp['K']}_seed{seed}")
        print_classification_report(preds, labels, f"SGC (K={hp['K']})")
        all_metrics.append(metrics)

        save_checkpoint(model, "sgc", seed=seed)
        save_run_results(metrics, history, model_name="sgc", seed=seed)
        print(f"  ✓ seed={seed}: acc={metrics['test_acc']:.4f}")

    if all_metrics:
        agg = aggregate_seeds(all_metrics)
        print(f"\n{'='*60}")
        print(f"  SGC AGGREGATE: acc={agg['test_acc_mean']:.4f} ± {agg['test_acc_std']:.4f}, "
              f"F1_macro={agg['f1_macro_mean']:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
