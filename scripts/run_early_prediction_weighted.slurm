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
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/nsfr:$PROJECT_ROOT/src/nudge:$PROJECT_ROOT/src/neumann:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

# 2. Paths Configuration (Defaults for Cluster)
DATASET_PATH="in/datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
if [ ! -f "$DATASET_PATH" ]; then
    DATASET_PATH="in/datasets/mimic_lazy_12_clean_with_interventions_corrected.npz"
fi
if [ ! -f "$DATASET_PATH" ]; then
    DATASET_PATH="/mnt/beegfs/cegbert/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
fi
if [ ! -f "$DATASET_PATH" ]; then
    DATASET_PATH="datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
fi
if [ ! -f "$DATASET_PATH" ]; then
    DATASET_PATH="/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
fi

CHECKPOINT="results/checkpoints/mimic/tune_mimic_cql"

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
    --tau-min 1 \
    --tau-max 36 \
    --tau-step 4 \
    --window-hours 12 \
    --epochs 10 \
    --n-models 100 \
    --output-dir "$OUTPUT_DIR"

echo "=== Weighted Sepsis Early Prediction Sweep Execution End ==="
date
