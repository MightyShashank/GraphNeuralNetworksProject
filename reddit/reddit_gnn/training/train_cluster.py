"""
ClusterGCN Training Loop.
Trains on METIS-partitioned subgraphs with optional diagonal enhancement.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tqdm import tqdm
from reddit_gnn.data.partition_cluster import apply_diagonal_enhancement
from reddit_gnn.training.utils import (
    EarlyStopping,
    get_scheduler,
    clip_gradients,
    log_epoch,
    measure_gpu_memory,
    reset_gpu_memory,
)
from reddit_gnn.training.train_neighbor import evaluate_neighbor  # noqa: F401 kept for ablations


@torch.no_grad()
def evaluate_cluster(model, data, device):
    """
    Full-graph evaluation via sparse matrix multiplication (SPMM).
    Same root cause as GraphSAINT: Reddit's 114M edges × 256 hidden = 116 GB
    when materializing the message passing [E, D] tensor.
    SparseTensor triggers GCNConv's SPMM path, keeping peak RAM ~4 GB.
    See evaluate_saint docstring for full explanation.
    """
    from torch_sparse import SparseTensor

    model.eval()
    model.to("cpu")

    try:
        N = data.num_nodes
        x = data.x.cpu()
        src, dst = data.edge_index[0].cpu(), data.edge_index[1].cpu()

        adj_t = SparseTensor(
            row=dst,  # destination
            col=src,  # source
            sparse_sizes=(N, N),
        )

        # edge_weight (diagonal enhancement) only applies during training batches
        out = model(x, adj_t)  # GCNConv uses SPMM

        val_out = out[data.val_mask.cpu()]
        val_y = data.y[data.val_mask].cpu()
        val_loss = F.cross_entropy(val_out, val_y).item()
        val_acc = (val_out.argmax(1) == val_y).float().mean().item()
    finally:
        model.to(device)

    return val_acc, val_loss


def train_cluster_gcn(
    model,
    cluster_data,
    data,
    optimizer,
    device,
    clusters_per_batch=20,
    lambda_val=0.1,
    max_epochs=50,
    patience=10,
    model_name="cluster_gcn",
    scheduler=None,
    verbose=True,
):
    """
    ClusterGCN training loop.

    Args:
        model: ClusterGCN model
        cluster_data: PyG ClusterData object (from METIS)
        data: Full graph data (for validation)
        clusters_per_batch: Number of clusters sampled per mini-batch
        lambda_val: Diagonal enhancement weight (0 = disabled)
    """
    from torch_geometric.loader import ClusterLoader

    train_loader = ClusterLoader(
        cluster_data,
        batch_size=clusters_per_batch,
        shuffle=True,
        num_workers=0,  # Must be 0 — forking after CUDA init causes crash
    )

    early_stop = EarlyStopping(patience=patience)
    if scheduler is None:
        scheduler = get_scheduler(optimizer)

    history = []
    best_val_acc = 0.0

    # ── Outer epoch bar ──────────────────────────────────────────────────────
    epoch_bar = tqdm(
        range(max_epochs),
        desc=f"[{model_name}] Training",
        unit="epoch",
        dynamic_ncols=True,
        disable=not verbose,
    )

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0
        n_batches = 0
        reset_gpu_memory(device)

        t0 = time.time()

        # ── Inner batch bar ──
        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch:3d} batches",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not verbose,
        )

        for batch in batch_bar:
            # Apply diagonal enhancement per mini-batch
            if lambda_val > 0:
                batch = apply_diagonal_enhancement(batch, lambda_val)

            batch = batch.to(device)
            optimizer.zero_grad()

            edge_weight = batch.edge_weight if hasattr(batch, "edge_weight") else None
            out = model(batch.x, batch.edge_index, edge_weight)

            # Only train on training nodes within this cluster batch
            if batch.train_mask.sum() == 0:
                continue

            loss = F.cross_entropy(
                out[batch.train_mask], batch.y[batch.train_mask]
            )
            loss.backward()
            clip_gradients(model)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (
                (out[batch.train_mask].argmax(1) == batch.y[batch.train_mask]).sum().item()
            )
            total_nodes += batch.train_mask.sum().item()
            n_batches += 1

            running_acc = total_correct / max(total_nodes, 1)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.4f}")

        batch_bar.close()

        epoch_time = time.time() - t0
        train_loss = total_loss / max(n_batches, 1)
        train_acc = total_correct / max(total_nodes, 1)
        gpu_mem = measure_gpu_memory(device)

        # Validation on CPU (full graph with correct val_mask)
        val_acc, val_loss = evaluate_cluster(model, data, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        entry = log_epoch(
            epoch, train_loss, val_loss, val_acc, epoch_time, gpu_mem, current_lr,
            extra={"train_acc": round(train_acc, 6)},
        )
        history.append(entry)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Update epoch bar
        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            tr_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}",
            vram=f"{gpu_mem:.0f}MB",
        )

        if early_stop.step(val_loss, model):
            epoch_bar.write(f"  [{model_name}] Early stopping at epoch {epoch}")
            break

    epoch_bar.close()
    early_stop.restore_best(model)

    if verbose:
        tqdm.write(f"  [{model_name}] Best val acc: {best_val_acc:.4f}")

    return history
