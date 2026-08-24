#!/bin/bash
#SBATCH --job-name=pyr_scaled_sweep
#SBATCH --output=results/logs/slurm/pyr_scaled_sweep_%j.out
#SBATCH --error=results/logs/slurm/pyr_scaled_sweep_%j.err
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cegbert@ncsu.edu

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

chmod +x scripts/run_pyrenees_6gpu_sweep.sh
./scripts/run_pyrenees_6gpu_sweep.sh
