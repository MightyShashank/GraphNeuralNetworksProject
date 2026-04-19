"""
Step 1A — Reddit Dataset Download and Verification (if the downloaded dataset is correct).
Downloads via PyG, verifies expected statistics.
"""

import torch
from torch_geometric.datasets import Reddit # Importing our built-in reddit dataset loader from Pytorch Geometric (it loads our ready-to-use graph object)
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..")) # Adding the parent directory to the search path, so we can directly import from there, hence the below reddit_gnn.config works

from reddit_gnn.config import REDDIT_RAW, EXPECTED_NODES, EXPECTED_EDGES, NUM_FEATURES, NUM_CLASSES, EXPECTED_TRAIN, EXPECTED_VAL, EXPECTED_TEST


def download_reddit(root: str = REDDIT_RAW) -> "torch_geometric.data.Data": # Type hint indicating that this function returns a PyTorch Geometric Data object (both in arg and return value), Returns PyG graph object (Data) though function actually returns both data, dataset (i love syntactic sugar you see)
    """Download Reddit dataset via PyG and verify statistics."""

    print(f"[1A] Downloading Reddit dataset to {root}...") 
    dataset = Reddit(root=root) # Loading the dataset (downloads from source if absent, if already downloaded then loads cached processed files)

    # Whenever you run/imoirt a .py file, py often compiles it to bytecode (.pyc) for faster future imports (see the _pychache__ in this folder) (we generally put this __pycache__ in .gitignore)
    data = dataset[0] # Extracting the graph object from the dataset (PyG datasets behave like indexable collections so Reddit dataset contains one giant graph so dataset[0] returns that graph)

    # ── Verify dataset properties ──
    checks = [
        ("Nodes", data.num_nodes, EXPECTED_NODES),
        ("Edges", data.num_edges, EXPECTED_EDGES),
        ("Node features", data.num_node_features, NUM_FEATURES),
        ("Classes", dataset.num_classes, NUM_CLASSES),
        ("Train nodes", data.train_mask.sum().item(), EXPECTED_TRAIN),
        ("Val nodes", data.val_mask.sum().item(), EXPECTED_VAL),
        ("Test nodes", data.test_mask.sum().item(), EXPECTED_TEST),
    ]

    all_pass = True
    for name, actual, expected in checks:
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_pass = False
        print(f"  {status} {name}: {actual:,} (expected {expected:,})")

    if not all_pass:
        print("\n⚠️  WARNING: Some dataset statistics don't match expected values!")
    else:
        print("\n✓ All dataset statistics verified successfully.")

    return data, dataset



if __name__ == "__main__":
    data, dataset = download_reddit()
    print(f"\nDataset loaded: {data}")
