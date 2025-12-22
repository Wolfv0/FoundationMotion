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

### Quickstart (demo.sh)
Run the end-to-end demo (batch processing with auto-resume):
```bash
bash scripts/submit_ranges.sh
```

This will start processing video data. Modify submit_range 0 60 to specify the range of videos to process — 0 is the starting index and 60 is the ending index. You can submit multiple jobs with different or even overlapping ranges; we handled all the rest for you. Just submit your jobs and adjust the start/end values as needed.

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

## Directory Structure
```
├── Videos/                 # Original videos directory
│   ├── name1.mp4
│   ├── name2.mp4
│   └── ...
│
├── Videos_crop/            # Preprocess
│   ├── name1.mp4
│   ├── name2.mp4
│   └── ...
│
├── Videos_crop_decode/     # Step-0
│   ├── name1/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── name2/
│   │   ├── 1.jpg
│   │   └── ...
│   └── ...
│
└── Videos_general_obj_det_finished/  # Step-1
    ├── name1.mp4
    ├── name2.mp4
    └── ...
```

### Current layout:
```
Videos
├── Videos/                          # Original input videos (used by demo.sh)
│   ├── 9fbfac6e733249496dfab5fd42cf329e.mp4
│   ├── diy_v_De7OXEsFTGY_frame000211__start_12547_end_12632.mp4
│   ├── diy_v_ECU5zeSI9eY_frame000140__start_8286_end_8376.mp4
│   ├── diy_v_tJ88nPS-Df8_frame000022__start_1013_end_1088.mp4
│   ├── Wolf_Data_2025-08-08_11.44.05.mp4
│   └── Wolf_Data_2025-08-08.mp4
├── video_general_obj_det_finished/  # Processed outputs (.mp4/.json and per-video dirs)
├── Videos_crop/
├── Videos_crop_decode/
├── videos_captions_gpt4o_mini/
├── videos_QAs_gpt4o_mini/
├── ignore/
```
