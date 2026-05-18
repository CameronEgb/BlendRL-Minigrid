#!/bin/bash
# cleanup_cluster.sh - Nukes all results on the cluster

DIRS=("logs" "datasets" "checkpoints" "tensorboard" "experiments" "plots")

echo "=== Nuking Result Directories ==="
for dir in "${DIRS[@]}"; do
    TARGET="results/$dir"
    if [ -d "$TARGET" ]; then
        echo "Clearing $TARGET..."
        rm -rf "$TARGET"/*
    else
        echo "Directory $TARGET does not exist, skipping."
    fi
done

echo "=== Cleanup Complete ==="
