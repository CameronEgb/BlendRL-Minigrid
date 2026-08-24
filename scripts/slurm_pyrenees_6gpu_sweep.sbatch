#!/bin/bash
#SBATCH --job-name=pyrenees_6gpu_sweep
#SBATCH --output=results/logs/slurm/pyrenees_6gpu_sweep_%j.out
#SBATCH --error=results/logs/slurm/pyrenees_6gpu_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:6
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cegbert@ncsu.edu

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

chmod +x scripts/run_pyrenees_6gpu_sweep.sh
./scripts/run_pyrenees_6gpu_sweep.sh
