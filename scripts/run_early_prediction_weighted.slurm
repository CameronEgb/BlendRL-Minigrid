#!/bin/bash
#SBATCH --job-name=sepsis_early_pred_weighted
#SBATCH --partition=rtx4060ti8g    # Default partition with idle GPUs
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cegbert@ncsu.edu

echo "=== Weighted Sepsis Early Prediction Sweep Execution Start ==="
echo "Node: $(hostname)"
date

# 1. Resolve PYTHONPATH
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

# 2. Paths Configuration (Defaults for Cluster)
DATASET_PATH=$(python3 -c "import sys; sys.path.insert(0, '.'); from src.pipeline.datasets import resolve_mimic_npz_path; print(resolve_mimic_npz_path('mimic_lazy_12_clean_with_interventions_corrected.npz'))" 2>/dev/null || echo "in/datasets/mimic/mimic_lazy_12_clean_with_interventions_corrected.npz")

CHECKPOINT="results/checkpoints/mimic/tune_mimic_all"

echo "Using Dataset Path: $DATASET_PATH"
echo "Using Checkpoint: $CHECKPOINT"

# 3. Create results directory for weighted experiment group
OUTPUT_DIR="results/plots/early_prediction/weighted_pr"
mkdir -p "$OUTPUT_DIR"
mkdir -p results/logs

# 4. Weighted loss & 4-metric early prediction sweep (AUC-ROC, AUPRC, F1-Max, F1-0.5)
$PROJECT_ROOT/venv/bin/python3 -u src/early_prediction/model.py \
    --checkpoint "$CHECKPOINT" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR"

echo "=== Weighted Sepsis Early Prediction Sweep Execution End ==="
date
