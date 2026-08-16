#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./cancel.sh [experiment_id]"
  exit 1
fi

EXP_ID=$1
IDS_FILE="results/slurm_ids/${EXP_ID}.txt"

# If exact file not found, try common mismatches or search in the slurm_ids directory
if [ ! -f "$IDS_FILE" ] && [ -d "results/slurm_ids" ]; then
    # Try searching for a file that contains the string
    SEARCH_FILE=$(ls results/slurm_ids/ 2>/dev/null | grep "$EXP_ID" | head -n 1)
    if [ -n "$SEARCH_FILE" ]; then
        IDS_FILE="results/slurm_ids/$SEARCH_FILE"
        echo "Found potential match: $IDS_FILE"
    fi
fi

if [ -f "$IDS_FILE" ]; then
  echo "Found job ID file: $IDS_FILE"
  JOB_IDS=$(cat "$IDS_FILE")
else
  echo "No job ID file found at $IDS_FILE. Falling back to squeue search..."
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
fi

if [ -z "$JOB_IDS" ]; then
  echo "No jobs found for experiment: $EXP_ID"
else
  echo "Canceling jobs for experiment '$EXP_ID':"
  # Echo each ID being canceled
  echo "$JOB_IDS" | xargs -I {} echo "  - Canceling job: {}"
  # Actually cancel them
  echo "$JOB_IDS" | xargs scancel
  
  # Remove the IDs file if it existed
  if [ -f "$IDS_FILE" ]; then
    rm "$IDS_FILE"
    echo "Removed $IDS_FILE"
  fi
  echo "Done."
fi
