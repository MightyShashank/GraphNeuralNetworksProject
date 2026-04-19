"""
conftest.py — pytest configuration for the Reddit GNN project.

Place this file at the `reddit/` directory root (parent of reddit_gnn/).
This ensures pytest adds the project root to sys.path so that
`from reddit_gnn.xxx import yyy` works without needing pip install.

Usage:
    cd /home/shashank/GNN/reddit
    pytest reddit_gnn/tests/test_baselines.py -v
"""

import sys
import os

# Ensure the `reddit/` directory (parent of `reddit_gnn/`) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
