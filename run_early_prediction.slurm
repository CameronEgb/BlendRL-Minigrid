#!/bin/bash
#SBATCH --job-name=sepsis_early_pred_dl
#SBATCH --partition=rtx4060ti8g    # Default partition with idle GPUs
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cegbert@ncsu.edu

echo "=== Sepsis Early Prediction Sweep Execution Start ==="
echo "Node: $(hostname)"
date

# 1. Resolve PYTHONPATH
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/nsfr:$PROJECT_ROOT/src/nudge:$PROJECT_ROOT/src/neumann:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

# 2. Paths Configuration (Defaults for Cluster)
DATASET_PATH="/mnt/beegfs/cegbert/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
if [ ! -f "$DATASET_PATH" ]; then
    DATASET_PATH="datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
fi
if [ ! -f "$DATASET_PATH" ]; then
    DATASET_PATH="/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
fi

CHECKPOINT="results/checkpoints/mimic/tune_mimic_cql"

echo "Using Dataset Path: $DATASET_PATH"
echo "Using Checkpoint: $CHECKPOINT"

# 3. Create results directory
mkdir -p results/plots/early_prediction
mkdir -p results/logs

# 4. Run early prediction sweep (LSTM vs Transformer, with vs without V(s))
# Trains and averages predictions of 100 models per configuration.
# Step size of 4 hours sweeps 9 points (1, 5, 9, ..., 33) to keep it fast while retaining smooth curves.
$PROJECT_ROOT/venv/bin/python3 src/septic_shock_early_prediction.py \
    --checkpoint "$CHECKPOINT" \
    --dataset-path "$DATASET_PATH" \
    --tau-min 1 \
    --tau-max 36 \
    --tau-step 4 \
    --epochs 10 \
    --n-models 100 \
    --output-dir "results/plots/early_prediction"

echo "=== Sepsis Early Prediction Sweep Execution End ==="
date
