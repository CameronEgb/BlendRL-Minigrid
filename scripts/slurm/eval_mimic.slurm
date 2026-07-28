#!/bin/bash
#SBATCH --job-name=eval_mimic_blendrl
#SBATCH --partition=rtx4060ti16g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=results/logs/slurm/%x_%j.out
#SBATCH --error=results/logs/slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cegbert@ncsu.edu

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH

echo "Starting early prediction evaluation for tune_mimic_blendrl_cql..."
$PROJECT_ROOT/venv/bin/python3 src/early_prediction_eval.py -e tune_mimic_blendrl_cql
echo "Evaluation complete."
