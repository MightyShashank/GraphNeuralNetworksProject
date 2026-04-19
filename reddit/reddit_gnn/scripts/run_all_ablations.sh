#!/usr/bin/env bash
# ============================================================
# run_all_ablations.sh — Master ablation runner
# Priority order from the implementation plan:
#   1. A4/C3 (graph vs features) + D4 (attention collapse)
#   2. A2/D3/E3 (oversmoothing)
#   3. Full ablation suite
#
# Usage (from project root, i.e. parent of reddit_gnn/):
#   bash reddit_gnn/scripts/run_all_ablations.sh
#   bash reddit_gnn/scripts/run_all_ablations.sh --priority-only
#   bash reddit_gnn/scripts/run_all_ablations.sh --model sage
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"   # reddit_gnn/
PARENT="$(dirname "$PROJECT_ROOT")"        # parent of reddit_gnn/

SEEDS="0 1 2"
PRIORITY_ONLY=false
MODEL_FILTER=""

# ── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --priority-only) PRIORITY_ONLY=true; shift ;;
    --model)         MODEL_FILTER="$2"; shift 2 ;;
    --seeds)         SEEDS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; shift ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
run_ablation() {
  local module="$1"; local ablations="$2"
  log "Running: python -m $module --ablation $ablations --seeds $SEEDS"
  (cd "$PARENT" && python -m "$module" --ablation $ablations --seeds $SEEDS)
}

echo "============================================================"
echo "  Reddit GNN — Ablation Suite"
echo "  Seeds: $SEEDS"
echo "  Priority only: $PRIORITY_ONLY"
echo "============================================================"

# ── PRIORITY 1: Highest research value ────────────────────────
log "=== PRIORITY 1: Core hypothesis ablations ==="

if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "sage" ]]; then
  run_ablation "reddit_gnn.ablations.run_sage_ablations"    "A4"
fi
if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "sgc" ]]; then
  run_ablation "reddit_gnn.ablations.run_sgc_ablations"     "C3"
fi
if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "gat" ]]; then
  run_ablation "reddit_gnn.ablations.run_gat_ablations"     "D4"
fi

if $PRIORITY_ONLY; then
  log "Priority-only mode: stopping after Priority 1."
  exit 0
fi

# ── PRIORITY 2: Oversmoothing depth ablations ─────────────────
log "=== PRIORITY 2: Oversmoothing ==="

if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "sage" ]]; then
  run_ablation "reddit_gnn.ablations.run_sage_ablations"    "A2"
fi
if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "gat" ]]; then
  run_ablation "reddit_gnn.ablations.run_gat_ablations"     "D3"
fi
if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "gatv2" ]]; then
  run_ablation "reddit_gnn.ablations.run_gatv2_ablations"   "E3"
fi

# ── PRIORITY 3: SGC ablations (fast, can run in parallel) ─────
log "=== PRIORITY 3: SGC ablations (fast) ==="
if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "sgc" ]]; then
  run_ablation "reddit_gnn.ablations.run_sgc_ablations"     "C1 C2"
fi

# ── PRIORITY 4: Remaining ablations ───────────────────────────
log "=== PRIORITY 4: Full ablation suite ==="

if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "sage" ]]; then
  run_ablation "reddit_gnn.ablations.run_sage_ablations"    "A1 A3 A5 A6"
fi
if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "saint" ]]; then
  run_ablation "reddit_gnn.ablations.run_saint_ablations"   "B1 B2 B3 B4"
fi
if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "gat" ]]; then
  run_ablation "reddit_gnn.ablations.run_gat_ablations"     "D1 D2"
fi
if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "gatv2" ]]; then
  run_ablation "reddit_gnn.ablations.run_gatv2_ablations"   "E1 E2"
fi
if [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "cluster" ]]; then
  run_ablation "reddit_gnn.ablations.run_cluster_ablations" "F1 F2 F3 F4"
fi

echo "============================================================"
log "ALL ABLATIONS COMPLETE"
echo "Results saved to: reddit_gnn/results/"
echo "Run notebooks to generate figures:"
echo "  jupyter notebook reddit_gnn/notebooks/"
echo "============================================================"
