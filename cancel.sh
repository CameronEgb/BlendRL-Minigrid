#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./cancel.sh [experiment_id]"
  exit 1
fi

EXP_ID=$1
# Check if squeue is available
if ! command -v squeue &> /dev/null; then
  echo "Error: 'squeue' command not found. Are you on the Slurm cluster?"
  exit 1
fi

USER_NAME=$(whoami)

# Find job IDs where the job name ends with _EXP_ID or is exactly EXP_ID
# Use -h to skip headers and %i %j for ID and Name. 
# We use a large width for %j to prevent truncation.
JOB_IDS=$(squeue -u "$USER_NAME" -h -o "%i %.100j" | grep -E "_${EXP_ID}$| ${EXP_ID}$" | awk '{print $1}')

if [ -z "$JOB_IDS" ]; then
  echo "No running jobs found for experiment: $EXP_ID"
else
  echo "Canceling jobs for experiment '$EXP_ID':"
  # Echo each ID being canceled
  echo "$JOB_IDS" | xargs -I {} echo "  - Canceling job: {}"
  # Actually cancel them
  echo "$JOB_IDS" | xargs scancel
  echo "Done."
fi
