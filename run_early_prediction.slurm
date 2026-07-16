#!/bin/bash
#SBATCH --job-name=sepsis_early_pred
#SBATCH --output=results/logs/sepsis_early_pred_%j.out
#SBATCH --error=results/logs/sepsis_early_pred_%j.err
#SBATCH --partition=rtx4060ti16g   # Default cluster partition, override as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --time=12:00:00            # Allow up to 12 hours (it will run much faster)
#SBATCH --mem=32G

echo "=== Sepsis Early Prediction Sweep Execution Start ==="
echo "Node: $(hostname)"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"
date

# 1. Resolve PYTHONPATH
export PYTHONPATH=src:$PYTHONPATH

# 2. Paths Configuration (Defaults for Cluster)
DATASET_PATH="/mnt/beegfs/cegbert/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
if [ ! -f "$DATASET_PATH" ]; then
    DATASET_PATH="datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
fi
if [ ! -f "$DATASET_PATH" ]; then
    DATASET_PATH="/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"
fi

CHECKPOINT="results/checkpoints/sepsis/mimic_cql/cql/0/best_model.ckpt"

echo "Using Dataset Path: $DATASET_PATH"
echo "Using Checkpoint: $CHECKPOINT"

# 3. Create results directory
mkdir -p results/plots/early_prediction
mkdir -p results/logs

# 4. Run early prediction sweep (rf, lr, lstm) across lead times 1 to 36 hours with a step of 2
python3 src/septic_shock_early_prediction.py \
    --checkpoint "$CHECKPOINT" \
    --dataset-path "$DATASET_PATH" \
    --models "rf,lr,lstm" \
    --tau-min 1 \
    --tau-max 36 \
    --tau-step 2 \
    --lstm-epochs 20 \
    --output-dir "results/plots/early_prediction"

echo "=== Sepsis Early Prediction Sweep Execution End ==="
date
