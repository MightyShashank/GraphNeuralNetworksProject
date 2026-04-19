"""
GraphSAINT Training Loop.
Key difference: normalization-corrected loss for unbiased gradients.
Loss = mean(per_node_loss * node_norm[train_mask])
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tqdm import tqdm
from reddit_gnn.training.utils import (
    EarlyStopping,
    get_scheduler,
    clip_gradients,
    log_epoch,
    measure_gpu_memory,
    reset_gpu_memory,
)
from reddit_gnn.training.train_neighbor import evaluate_neighbor


@torch.no_grad()
def evaluate_saint(model, data, device):
    """
    Full-graph evaluation via sparse matrix multiplication (SPMM).

    WHY SparseTensor instead of full-graph or NeighborLoader:
    - Full-graph CPU/GPU: GCN message passing materializes [E, D] tensor.
      114M edges × 256 hidden = 116 GB → OOM on both GPU and CPU.
    - NeighborLoader with bounded [25,10]: wrong for GCNConv because degree
      normalization D^{-1/2} is computed on the sampled subgraph, not the
      full graph. With Reddit's avg degree ~492, sampling 25 gives ~18x
      wrong normalization → garbage predictions (0.38 accuracy).
    - SparseTensor SPMM: computes A_norm @ H without ever materializing
      the [E, D] tensor. Peak RAM ~4 GB. Matches the original GraphSAINT
      paper's full-graph evaluation exactly.

    This is the standard approach per GraphSAINT paper + PyG documentation
    for large graphs where dense full-graph inference is impossible.
    """
    from torch_sparse import SparseTensor

    model.eval()
    model.to("cpu")

    try:
        N = data.num_nodes
        x = data.x.cpu()
        # edge_index: shape [2, E], row=src, col=dst
        src, dst = data.edge_index[0].cpu(), data.edge_index[1].cpu()

        # Build adj_t in PyG's expected format: adj_t[dst, src] = edge src→dst
        # GCNConv with SparseTensor calls gcn_norm() internally using sparse ops,
        # computing D^{-1/2}(A+I)D^{-1/2} without materializing [E, D].
        adj_t = SparseTensor(
            row=dst,  # destination (target)
            col=src,  # source
            sparse_sizes=(N, N),
        )

        out = model(x, adj_t)  # GCNConv uses SPMM, not scatter/gather over edges

        val_out = out[data.val_mask.cpu()]
        val_y = data.y[data.val_mask].cpu()
        val_loss = F.cross_entropy(val_out, val_y).item()
        val_acc = (val_out.argmax(1) == val_y).float().mean().item()
    finally:
        model.to(device)  # always move back to GPU

    return val_acc, val_loss

def train_saint(
    model,
    saint_loader,
    data,
    optimizer,
    device,
    max_epochs=30,
    patience=10,
    use_norm=True,
    model_name="graphsaint",
    scheduler=None,
    verbose=True,
):
    """
    GraphSAINT training loop.

    Args:
        model: GraphSAINTNet model
        saint_loader: GraphSAINT sampler loader
        data: Full graph data (for validation)
        optimizer: Optimizer
        device: Device
        use_norm: If True, apply normalization correction (B2 ablation)
        model_name: For logging
    """
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
        n_batches = 0
        reset_gpu_memory(device)

        t0 = time.time()

        # ── Inner batch bar ──
        batch_bar = tqdm(
            saint_loader,
            desc=f"  Epoch {epoch:3d} batches",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not verbose,
        )

        for batch in batch_bar:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)

            # Compute per-node loss
            loss_per_node = F.cross_entropy(
                out[batch.train_mask],
                batch.y[batch.train_mask],
                reduction="none",
            )

            if use_norm and hasattr(batch, "node_norm"):
                # Weight each node's loss by normalization correction
                norm_weights = batch.node_norm[batch.train_mask]
                loss = (loss_per_node * norm_weights).mean()
            else:
                # B2 ablation: no normalization correction
                loss = loss_per_node.mean()

            loss.backward()
            clip_gradients(model)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        batch_bar.close()

        epoch_time = time.time() - t0
        train_loss = total_loss / max(n_batches, 1)
        gpu_mem = measure_gpu_memory(device)

        # Validation on CPU (full graph with correct val_mask)
        val_acc, val_loss = evaluate_saint(model, data, device)

        # LR scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        entry = log_epoch(epoch, train_loss, val_loss, val_acc, epoch_time, gpu_mem, current_lr)
        history.append(entry)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Update epoch bar
        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}",
            vram=f"{gpu_mem:.0f}MB",
            lr=f"{current_lr:.2e}",
        )

        if early_stop.step(val_loss, model):
            epoch_bar.write(f"  [{model_name}] Early stopping at epoch {epoch}")
            break

    epoch_bar.close()
    early_stop.restore_best(model)

    if verbose:
        tqdm.write(f"  [{model_name}] Best val acc: {best_val_acc:.4f}")

    return history
