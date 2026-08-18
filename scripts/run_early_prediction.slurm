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
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

# 2. Paths Configuration (Defaults for Cluster)
DATASET_PATH=$(python3 -c "import sys; sys.path.insert(0, '.'); from src.pipeline.datasets import resolve_mimic_npz_path; print(resolve_mimic_npz_path('mimic_lazy_12_clean_with_interventions_corrected.npz'))" 2>/dev/null || echo "in/datasets/mimic/mimic_lazy_12_clean_with_interventions_corrected.npz")

CHECKPOINT="results/checkpoints/mimic/tune_mimic_all"

echo "Using Dataset Path: $DATASET_PATH"
echo "Using Checkpoint: $CHECKPOINT"

# 3. Create results directory
mkdir -p results/plots/early_prediction
mkdir -p results/logs

# 4. Fast early prediction sweep (LSTM vs Transformer, with vs without V(s))
# Uses n-models=5 and a step of 5 hours for quick plotting.
# Using python3 -u for unbuffered, live-updating logs.
$PROJECT_ROOT/venv/bin/python3 -u src/early_prediction/model.py \
    --checkpoint "$CHECKPOINT" \
    --dataset-path "$DATASET_PATH" \
    --exp-id "attention_volatility_sweep" \
    --output-dir "results/plots/early_prediction"

echo "=== Sepsis Early Prediction Sweep Execution End ==="
date
