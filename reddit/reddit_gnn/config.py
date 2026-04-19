"""
Global configuration for the Reddit GNN project.
All paths, hyperparameters, device setup, and constants.
"""

import os
import torch
import numpy as np
import random

# ─── Paths ──────────────────────────────────────────────────────────────────
# PROJECT_ROOT = reddit_gnn/ directory itself (self-contained)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data_store")
REDDIT_RAW = os.path.join(DATA_ROOT, "Reddit")
PREPROCESSED = os.path.join(DATA_ROOT, "preprocessed")
SGC_DIR = os.path.join(PREPROCESSED, "sgc")
CLUSTER_DIR = os.path.join(PREPROCESSED, "cluster")
SAINT_DIR = os.path.join(PREPROCESSED, "saint")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
EMBEDDINGS_DIR = os.path.join(RESULTS_ROOT, "embeddings")
CHECKPOINTS_DIR = os.path.join(RESULTS_ROOT, "checkpoints")
LOGS_DIR = os.path.join(RESULTS_ROOT, "logs")
FIGURES_DIR = os.path.join(RESULTS_ROOT, "figures")

# ─── Dataset constants ──────────────────────────────────────────────────────
NUM_CLASSES = 41
NUM_FEATURES = 602
EXPECTED_NODES = 232_965
EXPECTED_EDGES = 114_615_892
EXPECTED_TRAIN = 153_431
EXPECTED_VAL = 23_699
EXPECTED_TEST = 55_835

# ─── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── Reproducibility ────────────────────────────────────────────────────────
SEEDS = [0, 1, 2]


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Default hyperparameters ────────────────────────────────────────────────
DEFAULT_HPARAMS = {
    "graphsage": {
        "aggregator": "mean",
        "layers": 2,
        "hidden": 256,
        "dropout": 0.5,
        "skip": False,
        "norm": "batchnorm",
        "lr": 0.01,
        "weight_decay": 5e-4,
        "max_epochs": 50,
        "patience": 10,
        "num_neighbors": [25, 10],
        "batch_size": 1024,
    },
    "graphsaint": {
        "layers": 2,
        "hidden": 256,
        "dropout": 0.3,
        "sampler": "rw",
        "budget": 6000,
        "walk_length": 2,
        "num_steps": 30,
        "sample_coverage": 100,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "max_epochs": 30,
        "patience": 10,
    },
    "sgc": {
        "K": 2,
        "lr": 0.2,
        "weight_decay": 5e-4,
        "max_epochs": 100,
        "patience": 10,
    },
    "gat": {
        "layers": 2,
        "heads": 8,
        "hidden_per_head": 32,
        "attn_dropout": 0.3,
        "feat_dropout": 0.5,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "max_epochs": 50,
        "patience": 10,
        "num_neighbors": [25, 10],
        "batch_size": 1024,
    },
    "gatv2": {
        "layers": 2,
        "heads": 8,
        "hidden_per_head": 32,
        "attn_dropout": 0.3,
        "feat_dropout": 0.5,
        "share_weights": True,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "max_epochs": 50,
        "patience": 10,
        "num_neighbors": [25, 10],
        "batch_size": 1024,
    },
    "cluster_gcn": {
        "layers": 2,
        "hidden": 256,
        "dropout": 0.5,
        "lambda_val": 0.1,
        "num_parts": 1500,
        "clusters_per_batch": 20,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "max_epochs": 50,
        "patience": 10,
    },
}

# ─── WandB ──────────────────────────────────────────────────────────────────
WANDB_PROJECT = "reddit-gnn"
WANDB_ENABLED = True  # Set to False for CSV-only logging

# ─── Ensure directories exist ──────────────────────────────────────────────
for _dir in [DATA_ROOT, REDDIT_RAW, PREPROCESSED, SGC_DIR, CLUSTER_DIR,
             SAINT_DIR, RESULTS_ROOT, EMBEDDINGS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
             FIGURES_DIR,
             os.path.join(FIGURES_DIR, "01_baseline"),
             os.path.join(FIGURES_DIR, "02_ablation"),
             os.path.join(FIGURES_DIR, "03_visualisation"),
             os.path.join(FIGURES_DIR, "04_efficiency")]:
    os.makedirs(_dir, exist_ok=True)
