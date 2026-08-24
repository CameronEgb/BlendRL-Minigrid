#!/bin/bash
# ==============================================================================
# Pyrenees 6-GPU Overnight Optuna Hyperparameter Sweep Launcher
#
# Hardware: 6 x NVIDIA RTX 4060 Ti 16GB
# Execution: 2 concurrent workers per GPU (12 parallel streams total)
# Allocation:
#   - GPUs 0 & 1 (4 streams): Scaled Dueling ResNet (Neural Baseline)
#   - GPUs 2 & 3 (4 streams): Scaled BlendRL Human Logic + Dueling ResNet
#   - GPUs 4 & 5 (4 streams): Scaled BlendRL Neuro-Fuzzy CEW + Dueling ResNet
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Enable SQLite WAL mode on Optuna DB to allow high-concurrency locking across 12 workers
mkdir -p results/optuna
sqlite3 results/optuna/optuna_pyrenees_scaled.db "PRAGMA journal_mode=WAL;" 2>/dev/null || true

PYTHON_BIN="python3"
if [ -f "$PROJECT_ROOT/venv/bin/python3" ]; then
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python3"
fi

echo "========================================================================"
echo "          STARTING PYRENEES 6-GPU OVERNIGHT OPTUNA SWEEP"
echo "========================================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Python:       $PYTHON_BIN"
echo "Timestamp:    $(date)"
echo "------------------------------------------------------------------------"

PIDS=()

# --- GPUS 0 & 1: Scaled Neural Baseline (cql/dueling_resnet) ---
echo "[GPU 0 & 1] Launching 4 parallel workers for Scaled Dueling ResNet..."
for GPU_ID in 0 1; do
    for WORKER in 1 2; do
        CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN src/train.py \
            +experiment=pyrenees/tune_pyrenees_dueling_resnet \
            -m > "results/logs/slurm/optuna_dueling_resnet_gpu${GPU_ID}_w${WORKER}.log" 2>&1 &
        PIDS+=($!)
    done
done

# --- GPUS 2 & 3: Scaled Human Logic + Neural (blendrl_human_dueling_resnet) ---
echo "[GPU 2 & 3] Launching 4 parallel workers for BlendRL Human+ResNet..."
for GPU_ID in 2 3; do
    for WORKER in 1 2; do
        CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN src/train.py \
            +experiment=pyrenees/tune_pyrenees_blendrl_resnet \
            -m > "results/logs/slurm/optuna_blendrl_resnet_gpu${GPU_ID}_w${WORKER}.log" 2>&1 &
        PIDS+=($!)
    done
done

# --- GPUS 4 & 5: Scaled Neuro-Fuzzy CEW + Neural (blendrl_cew_dueling_resnet) ---
echo "[GPU 4 & 5] Launching 4 parallel workers for BlendRL CEW+ResNet..."
for GPU_ID in 4 5; do
    for WORKER in 1 2; do
        CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN src/train.py \
            +experiment=pyrenees/tune_pyrenees_cew_resnet \
            -m > "results/logs/slurm/optuna_cew_resnet_gpu${GPU_ID}_w${WORKER}.log" 2>&1 &
        PIDS+=($!)
    done
done

echo "------------------------------------------------------------------------"
echo "All 12 worker processes spawned successfully."
echo "Monitoring execution across 6 GPUs (PIDs: ${PIDS[*]})..."
echo "------------------------------------------------------------------------"

# Wait for all 12 workers to complete
for pid in "${PIDS[@]}"; do
    wait $pid || echo "Process $pid exited with status $?"
done

echo "========================================================================"
echo "          OPTUNA SWEEPS COMPLETED! EXTRACTING WINNING MODELS..."
echo "========================================================================"

# Run automated best model extraction, copy checkpoints, and run policy evaluation
$PYTHON_BIN scripts/extract_and_eval_best_pyrenees_models.py

echo "========================================================================"
echo "          ALL OVERNIGHT TASKS COMPLETE! (Finished: $(date))"
echo "========================================================================"
