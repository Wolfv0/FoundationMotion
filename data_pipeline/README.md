# FoundationMotion

## Env Setup

### Option 1  (Recommended)
```bash
conda env create -f environment.yml
conda activate fm
```

### Option 2

```bash
conda create -n fm python==3.10
conda activate fm
git clone https://github.com/sunrainyg/FoundationMotion.git && cd FoundationMotion

# Install cuda toolkit (to avoid error of CUDA_HOME not found)
conda install -c nvidia cuda-toolkit

# install SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..

# install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Downgrade pip to avoid "no module named torch" issue in installing GroundingDINO
pip install pip==22.3.1
pip install "setuptools>=62.3.0,<75.9"

# install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..

# install VGGT
pip install git+https://github.com/facebookresearch/vggt.git

# reinstall torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# install Additional Dependencies
conda install ffmpeg -c conda-forge
pip install accelerate imageio mmpose==0.24.0 mmcv==1.3.9 parso av moviepy pydantic openai
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```


## Download Pre-trained Models

### Main Models
Download and extract model files under the FoundationMotion folder:
```bash
wget http://fouheylab.eecs.umich.edu/~dandans/projects/spatial/models/models.zip --no-check-certificate
unzip models.zip
```

### GroundingDINO Weights
Download the GroundingDINO weights:
```bash
cd GroundingDINO
mkdir -p weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
```

### Environment Variables (Optional)
Set these environment variables if needed:
```bash
# Set Hugging Face cache locations
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_datasets_cache

# Set Hugging Face token (if using private models)
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

## Usage

### Quickstart
Run the end-to-end demo (batch processing with auto-resume):
```bash
bash scripts/submit_ranges.sh
```

This will start processing video data. Modify `submit_range 0 60` to specify the range of videos to process — 0 is the starting index and 60 is the ending index. You can submit multiple jobs with different or even overlapping ranges; we handled all the rest for you. Just submit your jobs and adjust the start/end values as needed.

### Detailed Pipeline

#### Step 1: Crop Videos
Randomly crop videos to 5-10 seconds and save them:
```bash
python crop_video.py --in_dir=/path/to/videos --out_dir=/path/to/videos_crop
```

After cropping, you'll have:
```
Videos_crop/
├── name1.mp4
├── name2.mp4
└── ...
```

#### Step 2: Decode Videos
Extract frames from the cropped videos:
```bash
python data_process/decode_video.py --clip_dir=/path/to/videos_crop
```

This creates the following structure with decoded frames:
```
Videos_crop_decode/
├── name1/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── name2/
│   ├── 1.jpg
│   └── ...
└── ...
```

#### Step 3: Object Detection
Run the object detection pipeline:
```bash
accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --mixed_precision fp16 \
    1_general_obj_det_v4.py \
    --video_dir=/path/to/Videos/ \
    --enable_camera_motion_detection \
    --chunk_idx -1
```

> **Important**: `--video_dir` should point to the original videos directory, not videos_crop.

#### Step 4: Generate Video Captions
Generate comprehensive video captions using GPT-4o based on video frames and motion information:
```bash
python 2_generate_caption.py \
    --model gpt4o_mini \
    --video_dir /path/to/video_general_obj_det_finished
```


#### Step 5: Generate Q&A Pairs
Generate multiple-choice Q&A pairs from video captions for motion understanding evaluation:
```bash
python 3_generate_QA.py \
    --model gpt4o_mini \
    --video_dir /path/to/video_general_obj_det_finished \
    --caption_dir /path/to/videos_captions_gpt4o_mini \
    --prompt_path prompts/caption_QA.prompt
```


## Directory Structure

Full pipeline output structure:
```
{DATA_ROOT}/
├── Videos/                              # Step 1: Original input videos
│   ├── name1.mp4
│   ├── name2.mp4
│   └── ...
│
├── Videos_crop/                         # Step 1: Cropped videos (5-10 seconds)
│   ├── name1.mp4
│   ├── name2.mp4
│   └── ...
│
├── Videos_crop_decode/                  # Step 2: Decoded frames
│   ├── name1/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── name2/
│   │   └── ...
│   └── ...
│
├── video_general_obj_det_finished/      # Step 3: Object detection outputs
│   ├── name1.mp4                        # Annotated video with bboxes
│   ├── name1.json                       # Motion info (bboxes, interactions)
│   ├── name1_camera_motion_skipped/     # (if camera motion detected)
│   ├── name2.mp4
│   ├── name2.json
│   └── ...
│
├── videos_captions_{model}/             # Step 4: Generated captions
│   ├── name1.json                       # Video caption JSON
│   ├── name2.json
│   └── ...
│
├── videos_QAs_{model}/                  # Step 5: Generated Q&A pairs
│   ├── name1.json                       # Multiple-choice QA JSON
│   ├── name2.json
│   └── ...
│
├── .chunk_assignments/                  # Parallel processing metadata
│   └── video_chunks_{N}.txt             # Stable chunk assignments
│
└── range_{start}_{end}_processing_summary.json  # Batch processing summaries
```

### Example layout:
```
Videos/
├── Videos/                              # Original input videos
│   ├── 9fbfac6e733249496dfab5fd42cf329e.mp4
│   ├── diy_v_De7OXEsFTGY_frame000211__start_12547_end_12632.mp4
│   └── Wolf_Data_2025-08-08.mp4
├── Videos_crop/                         # Cropped versions
├── Videos_crop_decode/                  # Extracted frames
├── video_general_obj_det_finished/      # Object detection results
├── videos_captions_gpt4o_mini/          # GPT-4o-mini captions
├── videos_QAs_gpt4o_mini/               # GPT-4o-mini Q&A pairs
└── range_0_60_processing_summary.json   # Batch processing log
```
