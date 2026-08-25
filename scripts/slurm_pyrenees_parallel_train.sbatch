#!/bin/bash
#SBATCH --job-name=pyr_parallel
#SBATCH --output=results/logs/slurm/pyrenees/parallel_train_%j.out
#SBATCH --error=results/logs/slurm/pyrenees/parallel_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cegbert@ncsu.edu

# ==============================================================================
# Pyrenees 11-Problem Parallel Training on RTX 4060 Ti 16GB
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

mkdir -p results/logs/slurm/pyrenees

PYTHON_BIN="python3"
if [ -f "$PROJECT_ROOT/venv/bin/python3" ]; then
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python3"
fi

echo "========================================================================"
echo "          PYRENEES 11-PROBLEM PARALLEL TRAINING JOB"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "Project Root:  $PROJECT_ROOT"
echo "Python:        $PYTHON_BIN"
echo "Timestamp:     $(date)"
echo "------------------------------------------------------------------------"

# 1. Train Scaled BlendRL Human Logic + Dueling ResNet (11 models in parallel)
echo "Training 11 BlendRL models in parallel on RTX 4060 Ti 16GB..."
$PYTHON_BIN scripts/train_pyrenees_all_problems.py \
    --method cql/blendrl_human_dueling_resnet \
    --exp-id pyrenees_multi_blendrl \
    --epochs 30 \
    --batch-size 512 \
    --parallel \
    --max-workers 11

echo "------------------------------------------------------------------------"
echo "Training complete! Exporting models to Pyrenees ITS..."
$PYTHON_BIN scripts/export_pyrenees_per_problem.py --verify

echo "========================================================================"
echo "          ALL TASKS COMPLETE! ($(date))"
echo "========================================================================"
