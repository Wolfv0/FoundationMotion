# Evaluation Scripts for FoundationMotion

This directory contains evaluation scripts for testing video understanding models on motion-related question answering benchmarks.

## Overview

Two evaluation scripts are provided:

1. **`gemini_motionbench.py`** - Evaluates Google Gemini models on the MotionBench dataset
2. **`vila_motionbench.py`** - Evaluates NVILA (VILA) models on custom video QA tasks

Both scripts process video files with multiple-choice questions and compute accuracy metrics.

---

## Requirements

```bash
pip install fire
pip install google-genai tqdm huggingface-hub

# You'll also need to set up Google Gemini API credentials:
export GOOGLE_API_KEY="your-api-key-here"
```


### For VILA Evaluation
```bash
# pip install llava  # NVILA/VILA model library
pip install -U https://github.com/NVlabs/VILA
```

---

## Usage

### 1. Gemini MotionBench Evaluation

Evaluate Gemini models on the MotionBench dataset with parallel processing support.

#### Basic Usage
```bash
python gemini_motionbench.py
```

#### Advanced Usage
```bash
python gemini_motionbench.py \
    --max_workers=8 \
    --model="gemini-2.5-flash" \
    --motionbench_base_path="/path/to/motionbench"
```


#### Output
Results are saved to `gemini_answer_motionbench_{model_name}.json` with the following structure:
```json
{
  "/path/to/video.mp4": [
    {
      "question": "What is happening in the video?",
      "prompt": "Question with choices...",
      "answer": "A",
      "output": "A",
      "score": 1
    }
  ],
  "total_acc": {
    "accuracy": 0.85,
    "correct": 85,
    "total": 100
  }
}
```

---

### 2. VILA MotionBench Evaluation

Evaluate NVILA models on custom video QA tasks.

#### Basic Usage
```bash
python vila_motionbench.py
```

#### Advanced Usage
```bash
python vila_motionbench.py \
    --task="av_hands_eval" \
    --base_dir="~/workspace/v2-dev" \
    --model_path="Efficient-Large-Model/NVILA-Lite-15B-Video"
```

#### Parameters
- `task` (str, default="av_hands_eval"): Task name used to construct directory paths
- `base_dir` (str, default="~/workspace/v2-dev"): Base directory containing task data
- `model_path` (str, default="Efficient-Large-Model/NVILA-Lite-15B-Video"): HuggingFace model path or local path to NVILA model

#### Expected Directory Structure
```
base_dir/
├── {task}/
│   ├── videos/          # MP4 video files
│   └── qa_shuffled/     # Corresponding JSON QA files
```

