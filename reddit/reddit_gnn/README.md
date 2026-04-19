# Reddit GNN — Node Classification on Reddit Dataset

> **GNN Node Classification** on the Reddit dataset (232K nodes, 114M edges, 41 subreddit classes)  
> **Models**: GraphSAGE · GraphSAINT · SGC · GAT · GATv2 · ClusterGCN  
> **Team**: Shashank Tippanavar · Swetha Murali · Ishita Kar

---

## Project Structure

Everything lives inside `reddit_gnn/` — this is the self-contained project root.

```
reddit_gnn/
├── config.py                              # Global paths, hyperparams, device, seeds
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Ignore data_store/, results/, caches
├── README.md                              # ← This file
│
├── data/                                  # Data loading & preprocessing
│   ├── download.py                        # Reddit dataset download + verification
│   ├── normalize.py                       # Z-score normalization (train-only stats)
│   ├── inspect_graph.py                   # Mask validation, degree stats, homophily
│   ├── precompute_sgc.py                  # SGC offline feature smoothing (K=1..5)
│   ├── partition_cluster.py               # METIS partitioning for ClusterGCN
│   └── loaders.py                         # NeighborLoader, GraphSAINT, ClusterLoader
│
├── models/                                # 6 GNN model definitions
│   ├── graphsage.py                       # GraphSAGE (mean/max/lstm/sum aggregator)
│   ├── graphsaint.py                      # GraphSAINT (GCN backbone)
│   ├── sgc.py                             # SGC (linear on precomputed A^K·X)
│   ├── gat.py                             # GAT (static multi-head attention)
│   ├── gatv2.py                           # GATv2 (dynamic attention)
│   └── cluster_gcn.py                     # ClusterGCN (GCN + diagonal enhancement)
│
├── training/                              # Training loops
│   ├── utils.py                           # EarlyStopping, LR scheduler, checkpointing
│   ├── train_neighbor.py                  # Shared loop for SAGE/GAT/GATv2
│   ├── train_saint.py                     # GraphSAINT (norm-corrected loss)
│   ├── train_sgc.py                       # SGC (logistic regression, no graph)
│   └── train_cluster.py                   # ClusterGCN (METIS subgraph training)
│
├── ablations/                             # 24 ablation study runners
│   ├── run_sage_ablations.py              # A1–A6: aggregator, depth, etc.
│   ├── run_saint_ablations.py             # B1–B4: sampler, budget, etc.
│   ├── run_sgc_ablations.py               # C1–C3: hops, norm, SGC vs MLP vs GCN
│   ├── run_gat_ablations.py               # D1–D4: heads, dropout, depth, analysis
│   ├── run_gatv2_ablations.py             # E1–E3: dynamic vs static, weights, depth
│   └── run_cluster_ablations.py           # F1–F4: clusters, lambda, batch, METIS
│
├── analysis/                              # Analysis & visualization modules
│   ├── oversmoothing.py                   # Per-layer embedding variance tracking
│   ├── expressivity.py                    # Aggregator collision detection (A1)
│   ├── attention_analysis.py              # Entropy, hub test, Kendall tau (D4/E1)
│   ├── visualisation.py                   # t-SNE/UMAP, 4 plot types
│   ├── efficiency.py                      # Timing, VRAM, throughput
│   └── homophily_degree.py                # 3×4 heatmap, regime classification
│
├── evaluation/                            # Evaluation & serialization
│   ├── metrics.py                         # Accuracy, F1, classification report
│   ├── structural_analysis.py             # Degree × homophily error analysis
│   └── serialize.py                       # JSON/CSV/NPY result saving
│
├── scripts/                               # Top-level runner scripts
│   ├── run_preprocessing.py               # Full preprocessing pipeline
│   ├── run_all_baselines.py               # Train all 6 models × 3 seeds
│   ├── run_sgc.py                         # SGC only
│   ├── run_graphsage.py                   # GraphSAGE only
│   ├── run_gat.py                         # GAT only
│   ├── run_gatv2.py                       # GATv2 only
│   ├── run_graphsaint.py                  # GraphSAINT only
│   └── run_cluster_gcn.py                 # ClusterGCN only
│
├── notebooks/                             # Analysis notebooks (runnable as scripts)
│   ├── 01_baseline_results.py             # Accuracy comparison, training curves
│   ├── 02_ablation_analysis.py            # Per-ablation bar charts, oversmoothing
│   ├── 03_visualisation.py                # t-SNE/UMAP embedding plots
│   └── 04_efficiency_report.py            # Timing, VRAM, throughput charts
│
├── data_store/                            # (auto-created, gitignored)
│   ├── Reddit/                            # Raw PyG download
│   └── preprocessed/                      # Normalized data, SGC features, METIS
│
└── results/                               # (auto-created, gitignored)
    ├── checkpoints/                       # Model weights
    ├── logs/                              # Training history CSVs
    ├── embeddings/                        # Extracted embeddings
    └── figures/                           # Generated plots
```

---

## Execution Order — Step by Step

### Step 0: Environment Setup

```bash
# Create and activate conda environment
conda create -n gnn python=3.10 -y
conda activate gnn

# Install PyTorch with CUDA (RTX A4500 system uses CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyG core (pure Python, no ABI concerns)
pip install torch-geometric
```

> ⚠️ **Critical — ABI Pinning:** `torch-scatter`, `torch-sparse`, and `pyg-lib` are
> compiled C++ binaries that **must exactly match your installed PyTorch version**.
> Using the wrong wheel silently installs but crashes at import with
> `OSError: undefined symbol`. Always use the matching wheel index below.

```bash
# Install PyG C++ extensions — wheel must match PyTorch + CUDA version exactly.
# Current pin: PyTorch 2.4.x + CUDA 12.1  →  torch-2.4.0+cu121
pip install torch-scatter torch-sparse pyg-lib \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# If you ever upgrade PyTorch, re-run the command above substituting:
#   torch-2.4.0+cu121  →  torch-<NEW_VER>+cu<CUDA_VER>
# e.g. for PyTorch 2.5.0 + CUDA 12.1: torch-2.5.0+cu121

# Install remaining Python dependencies
cd reddit_gnn/
pip install -r requirements.txt

# ── Verification (run all 3; any import error means ABI mismatch) ──
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)"
python -c "from torch_sparse import SparseTensor; print('torch_sparse: OK')"
python -c "from torch_scatter import scatter; print('torch_scatter: OK')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 1: Preprocessing (Run Once)

```bash
# From the parent directory of reddit_gnn/
python -m reddit_gnn.scripts.run_preprocessing
```

**What it does (in order):**
1. **1A** — Downloads Reddit dataset via PyG (~500MB download, auto-extracts)
2. **1B** — Z-score normalizes features using **train-only** statistics (prevents data leakage)
3. **1C/1D** — Validates split masks, computes degree stats, per-node homophily
4. **2A** — Precomputes SGC features `X_K = A^K·X` for K=1..5 (~2.8GB total)
5. **2B** — Runs METIS partitioning for 5 cluster configs (500, 1000, 1500, 3000, 6000)

**Expected output:** ~3.5GB in `reddit_gnn/data_store/`  
**Expected time:** ~15–30 minutes (METIS partitioning is the bottleneck)

### Step 2: Baseline Training

#### Option A — Train ALL 6 models at once (18 runs total: 6 models × 3 seeds)

```bash
python -m reddit_gnn.scripts.run_all_baselines
```

#### Option B — Train each model individually (recommended if resuming after a crash)

Each script trains only its model with 3 seeds by default. You can also pass
`--seeds 0` to run only a single seed (e.g., to re-run just the failed one).

```bash
# SGC  (~30 seconds total, already done)
python -m reddit_gnn.scripts.run_sgc
python -m reddit_gnn.scripts.run_sgc --seeds 0          # single seed

# GraphSAGE  (~45 min)
python -m reddit_gnn.scripts.run_graphsage
python -m reddit_gnn.scripts.run_graphsage --seeds 0    # single seed

# GAT  (~90 min)
python -m reddit_gnn.scripts.run_gat
python -m reddit_gnn.scripts.run_gat --seeds 1 2        # seeds 1 and 2 only

# GATv2  (~90 min)
python -m reddit_gnn.scripts.run_gatv2
python -m reddit_gnn.scripts.run_gatv2 --seeds 0

# GraphSAINT  (~60 min)
# Validation runs on CPU full-graph to correctly apply val_mask.
python -m reddit_gnn.scripts.run_graphsaint
python -m reddit_gnn.scripts.run_graphsaint --seeds 0

# ClusterGCN  (~30 min + ~5 min METIS cache)
# Validation runs on CPU full-graph to correctly apply val_mask.
python -m reddit_gnn.scripts.run_cluster_gcn
python -m reddit_gnn.scripts.run_cluster_gcn --seeds 0
```

> **Tip — Resuming after a crash:** If only one model failed (e.g. GraphSAINT),
> just run its individual script. Results from previously completed models are
> already saved in `results/` and do not need to be re-run.

**Expected output:** Checkpoints + metrics in `reddit_gnn/results/`  
**Expected time (per model):** SGC ~30s · GraphSAGE ~45min · GAT/GATv2 ~90min · GraphSAINT ~60min · ClusterGCN ~30min  
**Verification gate:** All 6 models must achieve >93% test accuracy

---

### Step 3: Ablation Studies (24 Studies, 5+ Variants Each)

Run ablations **per model** — they can be run independently or sequentially.

```bash
# GraphSAGE ablations (A1–A6)
python -m reddit_gnn.ablations.run_sage_ablations                    # All A1-A6
python -m reddit_gnn.ablations.run_sage_ablations --ablation A1 A4   # Specific ones
python -m reddit_gnn.ablations.run_sage_ablations --ablation A2 --seeds 0  # Single seed

# GraphSAINT ablations (B1–B4)
python -m reddit_gnn.ablations.run_saint_ablations
python -m reddit_gnn.ablations.run_saint_ablations --ablation B2     # Key: norm correction

# SGC ablations (C1–C3) — fastest, can run in parallel
python -m reddit_gnn.ablations.run_sgc_ablations
python -m reddit_gnn.ablations.run_sgc_ablations --ablation C3       # Key: SGC vs MLP vs GCN

# GAT ablations (D1–D4)
python -m reddit_gnn.ablations.run_gat_ablations
python -m reddit_gnn.ablations.run_gat_ablations --ablation D4       # Key: static attention analysis

# GATv2 ablations (E1–E3)
python -m reddit_gnn.ablations.run_gatv2_ablations
python -m reddit_gnn.ablations.run_gatv2_ablations --ablation E1     # Key: dynamic vs static

# ClusterGCN ablations (F1–F4)
python -m reddit_gnn.ablations.run_cluster_ablations
python -m reddit_gnn.ablations.run_cluster_ablations --ablation F2   # Key: diagonal enhancement
```

**Priority order** (from docx — run these first):
1. **A4/C3** — Graph structure vs features (highest research value)
2. **D4** — Static attention collapse detection
3. **A2/D3/E3** — Oversmoothing variance (captured during depth ablation)
4. **Remaining** — Run as time permits

**Expected time:** ~5–7 days total GPU time (single GPU)  
**Parallelization:** SGC ablations (C1-C3) don't need GPU for graph ops, can run simultaneously

---

### Step 4: Analysis Notebooks (Post-Training)

Run these **after** baselines and ablations are complete.

```bash
# 01: Baseline accuracy comparison + training curves
python -m reddit_gnn.notebooks.01_baseline_results

# 02: Ablation analysis — finds all results and generates comparison plots
python -m reddit_gnn.notebooks.02_ablation_analysis

# 03: Embedding visualization — t-SNE/UMAP (requires baseline checkpoints)
python -m reddit_gnn.notebooks.03_visualisation

# 04: Efficiency report — timing, VRAM, throughput charts
python -m reddit_gnn.notebooks.04_efficiency_report
```

**Output:** All figures saved to `reddit_gnn/results/figures/`

---

## Ablation Studies — Complete Reference

### GraphSAGE (A1–A6)

| ID | Study | Variants | Expected Finding |
|----|-------|----------|------------------|
| A1 | Aggregator Function | mean, max, lstm, sum | LSTM ≥ max > mean, sum captures multiset cardinality |
| A2 | Number of Layers | 1L, 2L, 3L, 4L, 5L | Peak at 2L, oversmoothing at 4-5L |
| A3 | Neighbor Sample Size | [5,5], [10,5], [15,10], [25,10], [50,25] | Diminishing returns past [25,10] |
| A4 | Structure vs Features | Full, FeaturesOnly, StructureOnly | Full >> FeaturesOnly >> StructureOnly |
| A5 | Skip Connections | NoSkip, ResAdd, ResConcat | Skip helps at depth ≥3 |
| A6 | Normalization | BatchNorm, LayerNorm, None | BatchNorm best for large batches |

### GraphSAINT (B1–B4)

| ID | Study | Variants | Expected Finding |
|----|-------|----------|------------------|
| B1 | Sampler Type | Node, Edge, RandomWalk | RW best for local coherence |
| B2 | Norm Correction | Enabled, Disabled | Disabled → bias toward hub nodes |
| B3 | Subgraph Budget | 2K, 4K, 6K, 10K, 15K | Diminishing returns past 6K |
| B4 | Walk Parameters | short/medium/long | Short walks best for Reddit |

### SGC (C1–C3)

| ID | Study | Variants | Expected Finding |
|----|-------|----------|------------------|
| C1 | Propagation Hops | K=1..5 | K=2 optimal, K≥4 oversmooths |
| C2 | Norm Scheme | Symmetric, Row, NoSelfLoop | Symmetric (Kipf) best |
| C3 | SGC vs MLP vs GCN | SGC, MLP, 2L-GCN | SGC ≈ GCN >> MLP (high homophily) |

### GAT (D1–D4)

| ID | Study | Variants | Expected Finding |
|----|-------|----------|------------------|
| D1 | Attention Heads | 1h–16h (constant 256 total) | 8h optimal tradeoff |
| D2 | Attn Dropout | 0.0–0.6 | 0.2-0.3 optimal |
| D3 | Depth | 1L–4L | Faster oversmoothing than SAGE |
| D4 | Static Analysis | Post-training analysis | Confirms attention collapse to GCN-like |

### GATv2 (E1–E3)

| ID | Study | Variants | Expected Finding |
|----|-------|----------|------------------|
| E1 | Dynamic vs Static | GATv2 vs GAT | GATv2 has lower Kendall τ (varying rankings) |
| E2 | Shared Weights | Shared vs Separate | Shared more parameter-efficient |
| E3 | Depth Comparison | 2-4L for both GAT/GATv2 | GATv2 tolerates depth better |

### ClusterGCN (F1–F4)

| ID | Study | Variants | Expected Finding |
|----|-------|----------|------------------|
| F1 | Cluster Count | 500–6000 | Accuracy drops linearly with edge retention |
| F2 | Diagonal Enhancement | λ=0..1 | Gain concentrates in boundary nodes |
| F3 | Clusters per Batch | 1–50 | 20 is sweet spot for convergence + speed |
| F4 | METIS vs Random | METIS, Random | METIS >> Random (community-aware) |

---

## Models — Architecture Summary

| Model | Paper | Layers | Hidden | Key Innovation |
|-------|-------|--------|--------|----------------|
| GraphSAGE | Hamilton et al., 2017 | 2 | 256 | Inductive neighborhood sampling + aggregation |
| GraphSAINT | Zeng et al., 2020 | 2 | 256 | Subgraph sampling with normalization correction |
| SGC | Wu et al., 2019 | 0 (linear) | — | `Y = softmax(A^K · X · θ)`, no nonlinearity |
| GAT | Veličković et al., 2018 | 2 | 8×32 | Static multi-head attention |
| GATv2 | Brody et al., 2022 | 2 | 8×32 | Dynamic attention (W after concat) |
| ClusterGCN | Chiang et al., 2019 | 2 | 256 | METIS partitioning + diagonal enhancement |

---

## Dataset — Reddit

| Property | Value |
|----------|-------|
| Nodes | 232,965 |
| Edges | 114,615,892 |
| Features | 602 (BoW + metadata) |
| Classes | 41 subreddits |
| Train/Val/Test | 153,431 / 23,699 / 55,835 |
| Mean degree | ~492 |
| Global homophily | ~0.93–0.95 |

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥16GB VRAM (tested on RTX A4500 24GB)
- **RAM**: ≥32GB system RAM
- **Disk**: ~15GB for data + checkpoints + results
- **GPU Time**: ~7h baselines + ~120h ablations (single GPU)
