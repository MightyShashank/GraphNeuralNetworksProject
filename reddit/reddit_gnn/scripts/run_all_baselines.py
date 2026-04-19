#!/usr/bin/env python3
"""
Master baseline runner — trains all 6 models with 3 seeds each.
Run: python scripts/run_all_baselines.py
"""

import sys
import os
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import (
    DEVICE, SEEDS, NUM_CLASSES, NUM_FEATURES, DEFAULT_HPARAMS, set_seed,
)
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.data.loaders import get_train_loader, get_val_loader
from reddit_gnn.training.utils import save_history, save_checkpoint, count_parameters
from reddit_gnn.training.train_neighbor import train_neighbor_sampled
from reddit_gnn.training.train_saint import train_saint
from reddit_gnn.training.train_sgc import train_sgc
from reddit_gnn.training.train_cluster import train_cluster_gcn
from reddit_gnn.evaluation.metrics import get_test_predictions, compute_all_metrics, print_classification_report
from reddit_gnn.evaluation.serialize import save_run_results


def run_graphsage_baseline(data, seed):
    """Train GraphSAGE baseline."""
    from reddit_gnn.models.graphsage import GraphSAGE

    hp = DEFAULT_HPARAMS["graphsage"]
    set_seed(seed)

    model = GraphSAGE(
        in_channels=NUM_FEATURES,
        hidden_channels=hp["hidden"],
        out_channels=NUM_CLASSES,
        num_layers=hp["layers"],
        dropout=hp["dropout"],
        aggregator=hp["aggregator"],
        skip=hp["skip"],
        norm=hp["norm"],
    ).to(DEVICE)

    print(f"\n{'='*60}")
    print(f"GraphSAGE Baseline (seed={seed}, params={count_parameters(model):,})")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    train_loader = get_train_loader(data, hp["num_neighbors"], hp["batch_size"])
    val_loader = get_val_loader(data, num_layers=hp["layers"], num_neighbors=hp["num_neighbors"], batch_size=1024)

    # --- TRAINING HAPPENS HERE ---
    # This function handles the entire PyTorch training loop for GraphSAGE:
    # forward passes, calculating loss, backpropagation, and early stopping.
    history = train_neighbor_sampled(
        model, train_loader, val_loader, optimizer, DEVICE,
        max_epochs=hp["max_epochs"], patience=hp["patience"],
        model_name="GraphSAGE",
    )

    # Evaluate
    preds, labels = get_test_predictions(model, data, DEVICE)
    metrics = compute_all_metrics(preds, labels, "graphsage", f"baseline_seed{seed}")
    print_classification_report(preds, labels, "GraphSAGE")

    # Save
    save_checkpoint(model, "graphsage", seed=seed)
    save_run_results(metrics, history, model_name="graphsage", seed=seed)

    return model, metrics, history


def run_sgc_baseline(data, seed):
    """Train SGC baseline."""
    hp = DEFAULT_HPARAMS["sgc"]
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"SGC Baseline (K={hp['K']}, seed={seed})")
    print(f"{'='*60}")

    model, history = train_sgc(
        K=hp["K"], data=data, device=DEVICE,
        max_epochs=hp["max_epochs"], lr=hp["lr"],
        weight_decay=hp["weight_decay"], patience=hp["patience"],
    )

    # Evaluate on precomputed features
    from reddit_gnn.config import SGC_DIR
    X_K = torch.load(os.path.join(SGC_DIR, f"reddit_sgc_K{hp['K']}.pt"), weights_only=False).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(X_K[data.test_mask.to(DEVICE)])
        preds = out.argmax(1).cpu().numpy()
        labels = data.y[data.test_mask].numpy()

    metrics = compute_all_metrics(preds, labels, "sgc", f"baseline_K{hp['K']}_seed{seed}")
    print_classification_report(preds, labels, f"SGC (K={hp['K']})")

    save_checkpoint(model, "sgc", seed=seed)
    save_run_results(metrics, history, model_name="sgc", seed=seed)

    return model, metrics, history


def run_gat_baseline(data, seed):
    """Train GAT baseline."""
    from reddit_gnn.models.gat import GAT

    hp = DEFAULT_HPARAMS["gat"]
    set_seed(seed)

    model = GAT(
        in_channels=NUM_FEATURES,
        out_channels=NUM_CLASSES,
        hidden_per_head=hp["hidden_per_head"],
        num_heads=hp["heads"],
        num_layers=hp["layers"],
        attn_dropout=hp["attn_dropout"],
        feat_dropout=hp["feat_dropout"],
    ).to(DEVICE)

    print(f"\n{'='*60}")
    print(f"GAT Baseline (seed={seed}, params={count_parameters(model):,})")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    train_loader = get_train_loader(data, hp["num_neighbors"], hp["batch_size"])
    val_loader = get_val_loader(data, num_layers=hp["layers"], num_neighbors=hp["num_neighbors"], batch_size=1024)

    history = train_neighbor_sampled(
        model, train_loader, val_loader, optimizer, DEVICE,
        max_epochs=hp["max_epochs"], patience=hp["patience"],
        model_name="GAT",
    )

    preds, labels = get_test_predictions(model, data, DEVICE)
    metrics = compute_all_metrics(preds, labels, "gat", f"baseline_seed{seed}")
    print_classification_report(preds, labels, "GAT")

    save_checkpoint(model, "gat", seed=seed)
    save_run_results(metrics, history, model_name="gat", seed=seed)

    return model, metrics, history


def run_gatv2_baseline(data, seed):
    """Train GATv2 baseline."""
    from reddit_gnn.models.gatv2 import GATv2

    hp = DEFAULT_HPARAMS["gatv2"]
    set_seed(seed)

    model = GATv2(
        in_channels=NUM_FEATURES,
        out_channels=NUM_CLASSES,
        hidden_per_head=hp["hidden_per_head"],
        num_heads=hp["heads"],
        num_layers=hp["layers"],
        attn_dropout=hp["attn_dropout"],
        feat_dropout=hp["feat_dropout"],
        share_weights=hp["share_weights"],
    ).to(DEVICE)

    print(f"\n{'='*60}")
    print(f"GATv2 Baseline (seed={seed}, params={count_parameters(model):,})")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    train_loader = get_train_loader(data, hp["num_neighbors"], hp["batch_size"])
    val_loader = get_val_loader(data, num_layers=hp["layers"], num_neighbors=hp["num_neighbors"], batch_size=1024)

    history = train_neighbor_sampled(
        model, train_loader, val_loader, optimizer, DEVICE,
        max_epochs=hp["max_epochs"], patience=hp["patience"],
        model_name="GATv2",
    )

    preds, labels = get_test_predictions(model, data, DEVICE)
    metrics = compute_all_metrics(preds, labels, "gatv2", f"baseline_seed{seed}")
    print_classification_report(preds, labels, "GATv2")

    save_checkpoint(model, "gatv2", seed=seed)
    save_run_results(metrics, history, model_name="gatv2", seed=seed)

    return model, metrics, history


def run_graphsaint_baseline(data, seed):
    """Train GraphSAINT baseline."""
    from reddit_gnn.models.graphsaint import GraphSAINTNet
    from reddit_gnn.data.loaders import get_saint_loader

    hp = DEFAULT_HPARAMS["graphsaint"]
    set_seed(seed)

    model = GraphSAINTNet(
        in_channels=NUM_FEATURES,
        hidden_channels=hp["hidden"],
        out_channels=NUM_CLASSES,
        num_layers=hp["layers"],
        dropout=hp["dropout"],
    ).to(DEVICE)

    print(f"\n{'='*60}")
    print(f"GraphSAINT Baseline (seed={seed}, params={count_parameters(model):,})")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    saint_loader = get_saint_loader(
        data, sampler_type=hp["sampler"], budget=hp["budget"],
        walk_length=hp["walk_length"], num_steps=hp["num_steps"],
        sample_coverage=hp["sample_coverage"],
    )

    history = train_saint(
        model, saint_loader, data, optimizer, DEVICE,
        max_epochs=hp["max_epochs"], patience=hp["patience"],
        model_name="GraphSAINT",
    )

    preds, labels = get_test_predictions(model, data, DEVICE, sparse_eval=True)
    metrics = compute_all_metrics(preds, labels, "graphsaint", f"baseline_seed{seed}")
    print_classification_report(preds, labels, "GraphSAINT")

    save_checkpoint(model, "graphsaint", seed=seed)
    save_run_results(metrics, history, model_name="graphsaint", seed=seed)

    return model, metrics, history


def run_cluster_gcn_baseline(data, seed):
    """Train ClusterGCN baseline."""
    from reddit_gnn.models.cluster_gcn import ClusterGCN
    from reddit_gnn.data.partition_cluster import prepare_cluster_gcn

    hp = DEFAULT_HPARAMS["cluster_gcn"]
    set_seed(seed)

    model = ClusterGCN(
        in_channels=NUM_FEATURES,
        hidden_channels=hp["hidden"],
        out_channels=NUM_CLASSES,
        num_layers=hp["layers"],
        dropout=hp["dropout"],
    ).to(DEVICE)

    print(f"\n{'='*60}")
    print(f"ClusterGCN Baseline (seed={seed}, params={count_parameters(model):,})")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    cluster_data = prepare_cluster_gcn(data, hp["num_parts"])

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

    save_checkpoint(model, "cluster_gcn", seed=seed)
    save_run_results(metrics, history, model_name="cluster_gcn", seed=seed)

    return model, metrics, history


def main():
    """Run all baselines."""
    print("=" * 60)
    print("REDDIT GNN — BASELINE TRAINING SUITE")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    data, _, _ = load_normalized_data()
    print(f"  Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")

    all_results = {}

    runners = [
        ("sgc", run_sgc_baseline),
        ("graphsage", run_graphsage_baseline),
        ("gat", run_gat_baseline),
        ("gatv2", run_gatv2_baseline),
        ("graphsaint", run_graphsaint_baseline),
        ("cluster_gcn", run_cluster_gcn_baseline),
    ]

    total_runs = len(runners) * len(SEEDS)
    overall_bar = tqdm(
        total=total_runs,
        desc="Overall progress",
        unit="run",
        dynamic_ncols=True,
    )

    for model_name, runner in runners:
        model_results = []
        for seed in SEEDS:
            overall_bar.set_description(f"[{model_name} seed={seed}]")
            try:
                _, metrics, _ = runner(data, seed)
                model_results.append(metrics)
                overall_bar.write(f"  ✓ {model_name} seed={seed}: acc={metrics['test_acc']:.4f}")
            except Exception as e:
                overall_bar.write(f"  ✗ {model_name} seed={seed} FAILED: {e}")
                import traceback
                traceback.print_exc()
            finally:
                overall_bar.update(1)

        if model_results:
            from reddit_gnn.evaluation.metrics import aggregate_seeds
            agg = aggregate_seeds(model_results)
            all_results[model_name] = agg
            overall_bar.write(
                f"  {model_name} AGGREGATE: "
                f"acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}"
            )

    overall_bar.close()

    # Final summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    for name, agg in all_results.items():
        print(f"  {name:15s}: acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}, "
              f"F1_macro={agg.get('f1_macro_mean',0):.4f}")


if __name__ == "__main__":
    main()
