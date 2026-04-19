# OOM & Crash Fixes — Reddit GNN Baselines

These are the fixes applied after running `python -m reddit_gnn.scripts.run_all_baselines`
and observing that GraphSAGE, GAT, GATv2, GraphSAINT, and ClusterGCN all failed.

---

## Bug 1 — GraphSAGE / GAT / GATv2 tried to allocate 200+ GiB

**Who crashed:** GraphSAGE, GAT, GATv2 (all 3 seeds each)

**Error message:**
```
CUDA out of memory. Tried to allocate 208.72 GiB.
```

**What was happening:**

Training was fine. The crash happened during *validation* at the end of each epoch.

The validation loader was created like this:
```python
num_neighbors = [-1, -1]  # -1 means "load ALL neighbors"
```

Reddit has ~492 average neighbors per node. With `-1` neighbors at 2 hops, fetching
even a small batch of 512 seed nodes would pull in millions of neighbor nodes, creating
a massive graph that needed 200+ GiB to hold in GPU memory.

**The fix:**

Changed the validation loader to use the same **bounded** neighbor sampling as training:
```python
# Before (BROKEN): unbounded - loads the entire neighborhood
val_loader = get_val_loader(data, num_layers=2)  # uses [-1, -1] internally

# After (FIXED): bounded - same fan-out as training
val_loader = get_val_loader(data, num_layers=2, num_neighbors=[25, 10], batch_size=1024)
```

**Files changed:** `data/loaders.py`, `scripts/run_all_baselines.py`

---

## Bug 2 — GraphSAINT tried to allocate 109 GiB

**Who crashed:** GraphSAINT (seed=0 — seeds 1 and 2 had a different crash, see Bug 3)

**Error message:**
```
CUDA out of memory. Tried to allocate 109.53 GiB.
```

**What was happening:**

GraphSAINT trains correctly on small random-walk subgraphs. But the validation
function (`evaluate_saint`) was doing this:

```python
# This loads the ENTIRE graph onto GPU at once
out = model(data.x, data.edge_index)  # data.edge_index has 114 MILLION edges
```

PyTorch Geometric's message passing then tried to allocate a dense matrix over
all 114 million edges, which needed 109 GiB.

**The fix:**

Replaced the full-graph validation function with the same mini-batched evaluation
that GraphSAGE uses (`evaluate_neighbor`). This processes only a few thousand
nodes at a time:

```python
# Before (BROKEN): loads all 114M edges at once
val_acc, val_loss = evaluate_saint(model, data, device)

# After (FIXED): processes validation nodes in small batches via NeighborLoader
val_acc, val_loss = evaluate_neighbor(model, val_loader, device)
```

ClusterGCN had the exact same bug (`evaluate_cluster` doing full-graph forward).
It was fixed the same way.

**Files changed:** `training/train_saint.py`, `training/train_cluster.py`, `scripts/run_all_baselines.py`

---

## Bug 3 — GraphSAINT seeds 1 & 2 + ClusterGCN: `CUDA error: initialization error` in DataLoader worker

**Who crashed:** GraphSAINT (seeds 1 & 2), ClusterGCN (all 3 seeds)

**Error messages:**
```
# GraphSAINT:
AssertionError: assert not data.edge_index.is_cuda

# ClusterGCN:
RuntimeError: CUDA error: initialization error (in DataLoader worker process 0)
```

**What was happening:**

Both errors come from the same root cause: **multiprocessing + CUDA don't mix** once
CUDA has already been initialized in the parent process.

When SGC and GraphSAGE ran earlier in the script, they initialized CUDA in the main
process. Later, when GraphSAINT or ClusterGCN created a `DataLoader` with
`num_workers=4`, Python used `fork()` to spawn child processes. These children
inherited the parent's broken/partial CUDA state.

- GraphSAINT's sampler detected that `data.edge_index` was already on CUDA (from
  earlier models leaving it there), and threw `AssertionError`.
- ClusterGCN's worker processes tried to start fresh CUDA contexts after inheriting
  the parent's state, which is a fatal conflict.

**The fix:**

Set `num_workers=0` across all loaders. This means data loading happens in the
**main process** — no forking, no CUDA inheritance problems. On Reddit, the bottleneck
is GPU computation not CPU data loading, so this has negligible impact on throughput.

```python
# Before (BROKEN): spawns 4 worker processes after CUDA is already initialized
num_workers = 4

# After (FIXED): everything runs in the main process
num_workers = 0
```

**Files changed:** `data/loaders.py` (all 5 loader functions)

---

## Bug 4 — `get_test_predictions` doing full-graph forward for evaluation

**Who was affected:** All models at test time (would have OOM'd the same way as Bug 2)

**What was happening:**

After training, `get_test_predictions` in `evaluation/metrics.py` was doing the same
dangerous thing as GraphSAINT's validator:
```python
out = model(data.x, data.edge_index)  # 114M edge full-graph forward
```

**The fix:**

Rewrote `get_test_predictions` to automatically create a bounded `NeighborLoader`
and evaluate in mini-batches:

```python
# Now dynamically creates a NeighborLoader with num_neighbors=[25, 10]
# and iterates over batches instead of materializing the entire graph
for batch in loader:
    out = model(batch.x, batch.edge_index)
    preds.append(out[:batch.batch_size].argmax(dim=1))
```

**Files changed:** `evaluation/metrics.py`

---

## Summary of all files changed

| File | What changed |
|------|-------------|
| `data/loaders.py` | `num_workers` → `0` everywhere; `get_val_loader` and `get_test_loader` now accept a `num_neighbors` argument |
| `training/train_saint.py` | Validation now uses `evaluate_neighbor` with a `val_loader` instead of full-graph `evaluate_saint` |
| `training/train_cluster.py` | Same as above — replaced `evaluate_cluster` with `evaluate_neighbor` |
| `evaluation/metrics.py` | `get_test_predictions` now uses a bounded `NeighborLoader` loop instead of full-graph forward |
| `scripts/run_all_baselines.py` | Passes bounded `val_loader` (with `[25, 10]` neighbors) to all models; creates and passes `val_loader` explicitly to GraphSAINT and ClusterGCN |
