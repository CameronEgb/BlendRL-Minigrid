#!/bin/bash
# scripts/ncshare_setup.sh — Environment setup script for NCShare cluster

set -e

WORK_DIR="/hpc/home/cegbert1/Offline-BlendRL"

echo "=================================================="
echo "=== Setting up NeSyRL on NCShare Cluster ==="
echo "=================================================="

echo "Working directory: $(pwd)"

# 2. Virtual environment setup
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment at ./venv..."
    python3 -m venv venv
else
    echo "Existing venv found at ./venv"
fi

# 3. Upgrade pip & install dependencies
echo "Installing / upgrading dependencies..."
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# 4. Create standard results directory structure
echo "Creating results directory hierarchy..."
mkdir -p results/logs/slurm
mkdir -p results/plots
mkdir -p results/checkpoints
mkdir -p results/datasets
mkdir -p results/optuna
mkdir -p results/hydra
mkdir -p results/slurm_ids
mkdir -p in/datasets/mimic

echo ""
echo "=================================================="
echo "=== Setup Complete ==="
echo "=================================================="
echo "To launch your mimic_tqn_all experiment, run:"
echo "  python3 run_pipeline.py mimic_tqn_all --local=false --partition=gpu ++recover=true"
echo ""
