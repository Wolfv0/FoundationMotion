set -euo pipefail

source scripts/config.sh

# Ensure caches and tokens are exported
export HF_HOME HF_DATASETS_CACHE HUGGING_FACE_HUB_TOKEN

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV_MAIN"
set -u

cd "$PROJECT_ROOT"

mkdir -p "$LOGS_DIR"

# Derive paths from config
VIDEO_DIR="${VIDEOS_DIR}"
BASE_VIDEOS_DIR="${DATA_ROOT}"
MODEL="${MODEL:-gpt4o_mini}"

echo "Processing range [${START}:${END}) from ${VIDEO_DIR} -> ${BASE_VIDEOS_DIR} using model=${MODEL}"

python process_video_range.py \
  --video_dir "${VIDEO_DIR}" \
  --base_videos_dir "${BASE_VIDEOS_DIR}" \
  --start_idx "${START}" \
  --end_idx "${END}" \
  --model "${MODEL}"

