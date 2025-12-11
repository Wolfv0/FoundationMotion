#!/bin/bash

# Centralized configuration for paths and common environment settings
# Source this file from scripts to avoid redefining paths.

# Absolute paths for data on this machine
####     hardcoded
export DATA_ROOT="/path/to/Videos"
######################
export VIDEOS_DIR="$DATA_ROOT/Videos"
export VIDEOS_CROP_DIR="$DATA_ROOT/Videos_crop"
export VIDEOS_DECODE_DIR="$DATA_ROOT/Videos_crop_decode"
export OUTPUT_GENERAL_DET_DIR="$DATA_ROOT/video_general_obj_det_finished"
export CAPTION_DIR="$DATA_ROOT/videos_captions_gpt4o_mini"

# Where to put generated batch scripts
export SCRIPTS_OUTPUT_DIR="batch_scripts"

# Project root and prompt paths

####     hardcoded
export PROJECT_ROOT="/path/to/FoundationMotion"
######################
export PROMPT_QA_PATH="$PROJECT_ROOT/prompts/caption_QA.prompt"
export PROMPT_CAPTION_PATH="$PROJECT_ROOT/prompts/video_caption_1K_general.prompt"
export LOGS_DIR="$PROJECT_ROOT/logs"

# Legacy compatibility (some scripts might still use PROMPT_PATH)
export PROMPT_PATH="$PROMPT_QA_PATH"

# Conda setup
####     hardcoded
export CONDA_SH="/path/to/miniconda3/etc/profile.d/conda.sh"
######################
export CONDA_ENV_MAIN="fm"

# HuggingFace caches and token
####     hardcoded
export HF_HOME="/path/to/hf_cache"
export HF_DATASETS_CACHE="/path/to/hf_cache"
export HUGGING_FACE_HUB_TOKEN="your_hf_token"
######################


