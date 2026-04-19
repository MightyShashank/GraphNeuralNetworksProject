"""
Evaluation Metrics — Accuracy, F1, classification report, aggregation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


@torch.no_grad()
def get_test_predictions(model, data_or_loader, device, model_type="default", sparse_eval=False):
    """
    Get predictions on the test set.

    Args:
        model_type: "sgc" for precomputed-feature models.
        sparse_eval: True for GCNConv-based models (GraphSAINT, ClusterGCN).
                     Uses SparseTensor SPMM to avoid the 116 GB OOM that
                     occurs with dense full-graph or NeighborLoader inference.
                     False (default) for SAGEConv/GATConv — safe with bounded
                     NeighborLoader because mean/sum aggregation isn't affected
                     by global degree unlike GCNConv's D^{-1/2} A D^{-1/2}.
    """
    model.eval()

    # ── SGC ────────────────────────────────────────────────────────────────────
    if model_type == "sgc":
        data = data_or_loader
        out = model(data.x.to(device))
        preds = out[data.test_mask].argmax(dim=1).cpu().numpy()
        labels = data.y[data.test_mask].cpu().numpy()
        return preds, labels

    # ── GCNConv models: SparseTensor SPMM (GraphSAINT, ClusterGCN) ────────────
    if sparse_eval:
        from torch_sparse import SparseTensor
        data = data_or_loader
        model.to("cpu")
        try:
            N = data.num_nodes
            src, dst = data.edge_index[0].cpu(), data.edge_index[1].cpu()
            adj_t = SparseTensor(row=dst, col=src, sparse_sizes=(N, N))
            out = model(data.x.cpu(), adj_t)
            preds = out[data.test_mask.cpu()].argmax(1).numpy()
            labels = data.y[data.test_mask].cpu().numpy()
        finally:
            model.to(device)
        return preds, labels

    # ── SAGEConv / GATConv: bounded NeighborLoader ─────────────────────────────
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    if isinstance(data_or_loader, Data):
        loader = NeighborLoader(
            data_or_loader,
            num_neighbors=[25, 10],
            batch_size=1024,
            input_nodes=data_or_loader.test_mask,
            shuffle=False,
            num_workers=0,
        )
    else:
        loader = data_or_loader

    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        bs = batch.batch_size
        all_preds.append(out[:bs].argmax(dim=1).cpu())
        all_labels.append(batch.y[:bs].cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()



def compute_all_metrics(preds, labels, model_name="", run_id=""):
    """Compute comprehensive evaluation metrics."""
    metrics = {
        "model": model_name,
        "run_id": run_id,
        # Primary metric
        "test_acc": accuracy_score(labels, preds),
        # Class-averaged metrics
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_micro": f1_score(labels, preds, average="micro"),
        # Per-class F1 (41 values)
        "f1_per_class": f1_score(labels, preds, average=None).tolist(),
    }
    return metrics


def print_classification_report(preds, labels, model_name=""):
    """Print sklearn classification report."""
    print(f"\n{'='*60}")
    print(f"Classification Report: {model_name}")
    print("=" * 60)
    print(classification_report(labels, preds, digits=4))


def compute_confusion_matrix(preds, labels):
    """Compute confusion matrix."""
    return confusion_matrix(labels, preds)


def aggregate_seeds(metrics_list):
    """
    Aggregate metrics across multiple seeds.
    Returns mean ± std for each numeric metric.
    """
    if not metrics_list:
        return {}

    aggregated = {}
    numeric_keys = ["test_acc", "f1_macro", "f1_weighted", "f1_micro"]

    for key in numeric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

    # Per-class F1 aggregation
    if "f1_per_class" in metrics_list[0]:
        per_class = np.array([m["f1_per_class"] for m in metrics_list])
        aggregated["f1_per_class_mean"] = per_class.mean(axis=0).tolist()
        aggregated["f1_per_class_std"] = per_class.std(axis=0).tolist()

    return aggregated
