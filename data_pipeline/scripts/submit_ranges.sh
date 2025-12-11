#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Load central config (provides DATA_ROOT, VIDEOS_DIR, CONDA_SH, CONDA_ENV_MAIN, etc.)
source "${PROJECT_ROOT}/scripts/config.sh"

# Optional override for model via env; default to gpt4o_mini
MODEL=${MODEL:-gpt4o_mini}

# Ensure logs directory exists from centralized config
mkdir -p "$LOGS_DIR"

submit_range() {
  local start=$1
  local end=$2
  echo "Submitting range [${start}:${end})"
  START=${start} END=${end} MODEL=${MODEL} \
    sbatch \
      --output="${LOGS_DIR}/%A_%a_range.out" \
      --error="${LOGS_DIR}/%A_%a_range.err" \
      "${SCRIPT_DIR}/process_range_template.sh"
  sleep 1
}

# Example ranges: [0:10), [10:20), [30:40), [40:50), [50:60)
# submit_range 0 11
submit_range 0 60

echo "All ranges submitted. Use 'squeue -u $USER' to check status."

