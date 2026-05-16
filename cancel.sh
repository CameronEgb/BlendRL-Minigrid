#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./cancel.sh [experiment_id]"
  exit 1
fi

EXP_ID=$1
USER_NAME=$(whoami)

# Find job IDs where the job name ends with _EXP_ID or is exactly EXP_ID
# Job names in run_pipeline_slurm.py look like: on_agent_EXP_ID or final_EXP_ID
JOB_IDS=$(squeue -u $USER_NAME -o "%i %j" | grep -E "_${EXP_ID}$| ${EXP_ID}$" | awk '{print $1}')

if [ -z "$JOB_IDS" ]; then
  echo "No running jobs found for experiment: $EXP_ID"
else
  echo "Canceling jobs for experiment '$EXP_ID':"
  echo "$JOB_IDS" | xargs -n 1 echo "  - Scanceling:"
  echo "$JOB_IDS" | xargs scancel
  echo "Done."
fi
