#!/usr/bin/env python3
"""
generate_plots.py — Generate all analysis plots from saved results.
Runs all 4 notebooks as scripts sequentially.

Run from parent of reddit_gnn/:
    python -m reddit_gnn.scripts.generate_plots
    python -m reddit_gnn.scripts.generate_plots --notebooks 01 03
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import RESULTS_ROOT

NOTEBOOKS = {
    "01": "Baseline Results — accuracy table, training curves, per-class F1",
    "02": "Ablation Analysis — all 24 studies, oversmoothing comparison",
    "03": "Embedding Visualization — t-SNE/UMAP (4 plot types × 6 models)",
    "04": "Efficiency Report — latency, VRAM, accuracy vs params",
}





def run_notebook_as_script(nb_id):
    """Execute a notebook via nbconvert (primary path) with a clear error fallback."""
    print(f"\n{'='*60}")
    print(f"  Notebook {nb_id}: {NOTEBOOKS[nb_id]}")
    print(f"{'='*60}")

    nb_path = os.path.join(
        os.path.dirname(__file__), "..", "notebooks", f"{nb_id}_{_nb_name(nb_id)}.ipynb"
    )

    if not os.path.exists(nb_path):
        print(f"  ✗ Notebook not found: {nb_path}")
        print(f"    Expected: reddit_gnn/notebooks/{nb_id}_{_nb_name(nb_id)}.ipynb")
        return

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
        "--execute", "--inplace", "--ExecutePreprocessor.timeout=3600", nb_path,
    ]
    print(f"  Executing: {os.path.basename(nb_path)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Notebook executed successfully")
    else:
        print(f"  ⚠️  nbconvert failed (returncode={result.returncode}):")
        print(f"    {result.stderr.strip()[-500:]}")
        print()
        print(f"  To run manually:")
        print(f"    jupyter nbconvert --to notebook --execute "
              f"reddit_gnn/notebooks/{nb_id}_{_nb_name(nb_id)}.ipynb --inplace")
        print(f"  Or open interactively:  jupyter notebook reddit_gnn/notebooks/")


def _nb_name(nb_id):
    names = {
        "01": "baseline_results",
        "02": "ablation_analysis",
        "03": "visualisation",
        "04": "efficiency_report",
    }
    return names[nb_id]


def main():
    parser = argparse.ArgumentParser(description="Generate all analysis plots")
    parser.add_argument("--notebooks", nargs="+", default=list(NOTEBOOKS.keys()),
                       choices=list(NOTEBOOKS.keys()),
                       help="Which notebooks to run (01 02 03 04)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Reddit GNN — Plot Generation")
    print(f"  Notebooks: {args.notebooks}")
    print(f"  Output: {RESULTS_ROOT}/figures/")
    print("=" * 60)

    for nb_id in args.notebooks:
        run_notebook_as_script(nb_id)

    print(f"\n{'='*60}")
    print("PLOT GENERATION COMPLETE")
    print(f"Figures saved to: {RESULTS_ROOT}/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
