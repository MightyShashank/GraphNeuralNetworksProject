"""
SGC Training Loop.
The simplest loop — no graph access during training.
Logistic regression on precomputed X_K features.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tqdm import tqdm
from reddit_gnn.config import SGC_DIR, NUM_CLASSES, DEVICE
from reddit_gnn.training.utils import (
    EarlyStopping,
    log_epoch,
    measure_gpu_memory,
    reset_gpu_memory,
)


def train_sgc(
    K,
    data,
    device=None,
    max_epochs=100,
    lr=0.2,
    weight_decay=5e-4,
    patience=10,
    sgc_dir=None,
    verbose=True,
):
    """
    Train SGC: linear classifier on precomputed X_K features.

    Args:
        K: Number of propagation hops (uses precomputed reddit_sgc_K{K}.pt)
        data: PyG Data object (need y, train_mask, val_mask)
        device: torch.device
    """
    if device is None:
        device = DEVICE
    if sgc_dir is None:
        sgc_dir = SGC_DIR

    # Load precomputed features
    feature_path = os.path.join(sgc_dir, f"reddit_sgc_K{K}.pt")
    X_K = torch.load(feature_path, weights_only=False).to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)

    if verbose:
        tqdm.write(f"  [SGC K={K}] Loaded features: {feature_path} → {X_K.shape}")

    # Simple logistic regression classifier
    from reddit_gnn.models.sgc import SGC

    model = SGC(X_K.shape[1], NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stop = EarlyStopping(patience=patience)

    history = []
    best_val_acc = 0.0

    # ── Epoch bar (SGC is full-batch, so no inner bar needed) ────────────────
    epoch_bar = tqdm(
        range(max_epochs),
        desc=f"[SGC K={K}] Training",
        unit="epoch",
        dynamic_ncols=True,
        disable=not verbose,
    )

    for epoch in epoch_bar:
        reset_gpu_memory(device)
        t0 = time.time()

        # Training
        model.train()
        optimizer.zero_grad()
        out = model(X_K[train_mask])
        loss = F.cross_entropy(out, y[train_mask])
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        train_acc = (out.argmax(1) == y[train_mask]).float().mean().item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_K[val_mask])
            val_loss = F.cross_entropy(val_out, y[val_mask]).item()
            val_acc = (val_out.argmax(1) == y[val_mask]).float().mean().item()

        epoch_time = time.time() - t0
        gpu_mem = measure_gpu_memory(device)

        entry = log_epoch(
            epoch, train_loss, val_loss, val_acc, epoch_time, gpu_mem,
            extra={"train_acc": round(train_acc, 6)},
        )
        history.append(entry)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Update bar every epoch (SGC is fast)
        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            tr_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}",
            t=f"{epoch_time*1000:.0f}ms",
        )

        if early_stop.step(val_loss, model):
            epoch_bar.write(f"  [SGC K={K}] Early stopping at epoch {epoch}")
            break

    epoch_bar.close()
    early_stop.restore_best(model)

    if verbose:
        tqdm.write(f"  [SGC K={K}] Best val acc: {best_val_acc:.4f}")

    return model, history
