"""
Integrated video processing function that combines all preprocessing steps.
This function processes a single video through all steps:
1. Crop video (5-10 seconds)
2. Decode video to frames
3. Object detection and motion analysis
4. Generate hand captions
5. Generate Q&A pairs

Usage:
    python process_single_video.py --video_path /path/to/video.mp4 --base_output_dir /path/to/output
"""

import os
import sys
import json
import glob
import random
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from existing scripts
from data_process.crop_video import get_video_duration, crop_video
from data_process.decode_video import decode_video, has_video_stream

def setup_environment():
    """Set up environment variables needed for the pipeline"""
    pass

def check_video_processing_status(video_path: str, base_videos_dir: str, model: str = "gpt4o_mini") -> dict:
    """
    Check which steps have already been completed for a video.
    Returns a dictionary indicating the status of each step.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    status = {
        "preprocess": False,
        "decode": False,
        "object_detection": False,
        "captions": False,
        "qa": False
    }
    
    # Check preprocess step
    cropped_video_path = os.path.join(base_videos_dir, "Videos_crop", os.path.basename(video_path))
    status["preprocess"] = os.path.exists(cropped_video_path)
    
    # Check decode step
    decode_dir = os.path.join(base_videos_dir, "Videos_crop_decode", video_name)
    if os.path.exists(decode_dir):
        frame_files = [f for f in os.listdir(decode_dir) if f.endswith('.jpg')]
        status["decode"] = len(frame_files) > 0
    
    # Check object detection step
    obj_det_video = os.path.join(base_videos_dir, "video_general_obj_det_finished", f"{video_name}.mp4")
    obj_det_json = os.path.join(base_videos_dir, "video_general_obj_det_finished", f"{video_name}.json")
    status["object_detection"] = os.path.exists(obj_det_video) and os.path.exists(obj_det_json)
    
    # Check captions step
    caption_file = os.path.join(base_videos_dir, f"videos_captions_{model}", f"{video_name}.json")
    status["captions"] = os.path.exists(caption_file)
    
    # Check Q&A step
    qa_file = os.path.join(base_videos_dir, f"videos_QAs_{model}", f"{video_name}.json")
    status["qa"] = os.path.exists(qa_file)
    
    return status

def step_preprocess(video_path: str, base_videos_dir: str) -> str:
    """
    Step 0: Crop video to 5-10 seconds
    Returns: path to cropped video
    """
    print(f"=== STEP PREPROCESS: Cropping video ===")
    
    video_name = os.path.basename(video_path)
    videos_crop_dir = os.path.join(base_videos_dir, "Videos_crop")
    cropped_video_path = os.path.join(videos_crop_dir, video_name)
    
    # Create output directory
    os.makedirs(videos_crop_dir, exist_ok=True)
    
    # Check if already processed
    if os.path.exists(cropped_video_path):
        print(f"✓ Cropped video already exists, skipping: {video_name}")
        return cropped_video_path
    
    # Get video duration
    duration = get_video_duration(video_path)
    
    if duration <= 5.0:
        # If video is shorter than 5s, copy it directly
        print(f"Video {video_name} is {duration:.2f}s (≤5s), copying directly")
        shutil.copy2(video_path, cropped_video_path)
    else:
        # Choose a random duration between 5-10s
        crop_duration = random.uniform(5.0, min(10.0, duration))
        
        # Calculate start time (try to position in the middle)
        max_start = duration - crop_duration
        middle_point = duration / 2 - crop_duration / 2
        
        # Add some randomness but keep it near the middle
        variation = min(duration * 0.2, max_start / 2)
        start_time = max(0, min(max_start, middle_point + random.uniform(-variation, variation)))
        
        print(f"Video {video_name} is {duration:.2f}s, cropping {crop_duration:.2f}s from position {start_time:.2f}s")
        crop_video(video_path, cropped_video_path, start_time, crop_duration)
    
    return cropped_video_path

def step_0_decode(cropped_video_path: str, base_videos_dir: str) -> str:
    """
    Step 0: Decode video to frames
    Returns: path to decoded frames directory
    """
    print(f"=== STEP 0: Decoding video to frames ===")
    
    video_name = os.path.splitext(os.path.basename(cropped_video_path))[0]
    videos_crop_decode_dir = os.path.join(base_videos_dir, "Videos_crop_decode")
    decode_dir = os.path.join(videos_crop_decode_dir, video_name)
    
    # Create base decode directory
    os.makedirs(videos_crop_decode_dir, exist_ok=True)
    
    # Check if already processed (has frames)
    if os.path.exists(decode_dir):
        frame_files = [f for f in os.listdir(decode_dir) if f.endswith('.jpg')]
        if len(frame_files) > 0:
            print(f"✓ Frames already decoded ({len(frame_files)} frames), skipping: {video_name}")
            return decode_dir
    
    # Decode video to frames
    decode_video(cropped_video_path, decode_dir, video_name)
    
    return decode_dir

def step_1_object_detection(video_path: str, base_videos_dir: str) -> str:
    """
    Step 1: Object detection and motion analysis
    Returns: path to processed video with object detection results
    """
    print(f"=== STEP 1: Object detection and motion analysis ===")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_dir = os.path.join(base_videos_dir, "video_general_obj_det_finished")
    
    # Check if already processed
    output_video_path = os.path.join(output_video_dir, f"{video_name}.mp4")
    output_json_path = os.path.join(output_video_dir, f"{video_name}.json")
    
    if os.path.exists(output_video_path) and os.path.exists(output_json_path):
        print(f"✓ Object detection already completed, skipping: {video_name}")
        return output_video_dir
    
    # Set up SAM2 and GroundingDINO
    setup_cmd = """
    cd sam2 && python setup.py build_ext --inplace && cd ..
    cd GroundingDINO && python setup.py build_ext --inplace && cd ..
    """
    
    # Use the existing Videos directory or create it in the base_videos_dir
    videos_dir = os.path.join(base_videos_dir, "Videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Copy the original video to the Videos directory if not already there
    video_name_with_ext = os.path.basename(video_path)
    temp_video_path = os.path.join(videos_dir, video_name_with_ext)
    if not os.path.exists(temp_video_path):
        shutil.copy2(video_path, temp_video_path)
    
    # Run object detection - it will output to video_general_obj_det_finished in the same base directory
    # Use single_video_name to process only the current video
    cmd = f"""
    accelerate launch \
        --num_processes 1 \
        --num_machines 1 \
        --machine_rank 0 \
        --mixed_precision fp16 \
        1_general_obj_det_v4.py \
        --video_dir={base_videos_dir} \
        --enable_camera_motion_detection \
        --single_video_name={video_name}
    """
    
    print(f"Running object detection command: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        raise RuntimeError(f"Object detection failed with return code {result.returncode}")
    
    return output_video_dir

def step_2_hand_caption(video_dir: str, base_videos_dir: str, model: str = "gpt4o_mini", video_names: list = None) -> str:
    """
    Step 2: Generate hand captions
    Returns: path to captions directory
    """
    print(f"=== STEP 2: Generating hand captions ===")
    
    # The output should be in videos_captions_{model} directory in the base_videos_dir
    captions_dir = os.path.join(base_videos_dir, f"videos_captions_{model}")
    
    # Check if captions already exist for this video
    # We need to extract the video name from the video_dir path
    # Since video_dir is the base directory, we need to check if any video files have captions
    video_files = []
    if os.path.exists(video_dir):
        import glob
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    # For now, we'll run the command and let the script handle the resume logic
    # The 2_generate_caption.py already has built-in resume functionality
    
    if video_names:
        # Use provided video names
        video_names_str = ' '.join(video_names)
        cmd = f"""
        python 2_generate_caption.py \
            --model={model} \
            --video_dir={video_dir} \
            --video_names {video_names_str}
        """
    else:
        # Extract video name from video_dir path (fallback)
        video_name = os.path.basename(video_dir).replace('.mp4', '')
        cmd = f"""
        python 2_generate_caption.py \
            --model={model} \
            --video_dir={video_dir} \
            --video_names {video_name}
        """
    
    print(f"Running hand caption command: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        raise RuntimeError(f"Hand caption generation failed with return code {result.returncode}")
    
    return captions_dir

def step_3_generate_qa(video_dir: str, caption_dir: str, base_videos_dir: str, model: str = "gpt4o_mini", video_names: list = None) -> str:
    """
    Step 3: Generate Q&A pairs
    Returns: path to Q&A directory
    """
    print(f"=== STEP 3: Generating Q&A pairs ===")
    
    # The output should be in videos_QAs_{model} directory in the base_videos_dir
    qa_dir = os.path.join(base_videos_dir, f"videos_QAs_{model}")
    
    # The 3_generate_QA.py script already has built-in resume functionality
    # It checks if QA files already exist and skips them
    
    # prompt_path = "/data/vision/torralba/selfmanaged/isola/u/yulu/cmar/FoundationMotion22/prompts/caption_QA.prompt"
    prompt_path = "prompts/caption_QA.prompt"
    
    if video_names:
        # Use provided video names
        video_names_str = ' '.join(video_names)
        cmd = f"""
        python 3_generate_QA.py \
            --model={model} \
            --video_dir={video_dir} \
            --caption_dir={caption_dir} \
            --prompt_path={prompt_path} \
            --video_names {video_names_str}
        """
    else:
        # Extract video name from video_dir path (fallback)
        video_name = os.path.basename(video_dir).replace('.mp4', '')
        cmd = f"""
        python 3_generate_QA.py \
            --model={model} \
            --video_dir={video_dir} \
            --caption_dir={caption_dir} \
            --prompt_path={prompt_path} \
            --video_names {video_name}
        """
    
    print(f"Running Q&A generation command: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        raise RuntimeError(f"Q&A generation failed with return code {result.returncode}")
    
    return qa_dir

def process_single_video(video_path: str, base_videos_dir: str, model: str = "gpt4o_mini", force_reprocess: bool = False) -> dict:
    """
    Process a single video through all preprocessing steps with auto-resume functionality.
    All outputs will be stored in the unified directory structure under base_videos_dir.
    
    Args:
        video_path: Path to the input video file
        base_videos_dir: Base Videos directory (like /path/to/Videos) where all outputs will be stored
        model: Model to use for caption and Q&A generation
        force_reprocess: If True, reprocess all steps even if outputs exist
    
    Returns:
        Dictionary containing paths to all output files/directories
    """
    print(f"Processing video: {video_path}")
    print(f"Base videos directory: {base_videos_dir}")
    
    # Set up environment
    setup_environment()
    
    # Create base videos directory
    os.makedirs(base_videos_dir, exist_ok=True)
    
    # Get video name for tracking
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Check current processing status
    if not force_reprocess:
        status = check_video_processing_status(video_path, base_videos_dir, model)
        completed_steps = [step for step, done in status.items() if done]
        if completed_steps:
            print(f"📋 Resume check - Already completed steps: {', '.join(completed_steps)}")
        
        # If all steps are completed, skip processing
        if all(status.values()):
            print(f"✓ Video {video_name} is already fully processed. Skipping.")
            return {
                "video_name": video_name,
                "original_video": video_path,
                "base_videos_dir": base_videos_dir,
                "status": "already_completed",
                "completed_steps": completed_steps
            }
    else:
        print(f"🔄 Force reprocessing enabled - will reprocess all steps")
    
    results = {
        "video_name": video_name,
        "original_video": video_path,
        "base_videos_dir": base_videos_dir
    }
    
    try:
        # Step Preprocess: Crop video -> Videos_crop/
        cropped_video = step_preprocess(video_path, base_videos_dir)
        results["cropped_video"] = cropped_video
        
        # Step 0: Decode video to frames -> Videos_crop_decode/
        decoded_frames_dir = step_0_decode(cropped_video, base_videos_dir)
        results["decoded_frames"] = decoded_frames_dir
        
        # Step 1: Object detection -> video_general_obj_det_finished/
        obj_det_dir = step_1_object_detection(video_path, base_videos_dir)
        results["object_detection"] = obj_det_dir
        
        # Step 2: Hand captions -> videos_captions_{model}/
        captions_dir = step_2_hand_caption(obj_det_dir, base_videos_dir, model)
        results["captions"] = captions_dir
        
        # Step 3: Q&A generation -> videos_QAs_{model}/
        qa_dir = step_3_generate_qa(obj_det_dir, captions_dir, base_videos_dir, model)
        results["qa"] = qa_dir
        
        results["status"] = "success"
        print(f"✓ Successfully processed video: {video_name}")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        print(f"✗ Failed to process video {video_name}: {e}")
        raise
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Process a single video through all preprocessing steps with auto-resume')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to the input video file')
    parser.add_argument('--base_videos_dir', type=str, required=True,
                        help='Base Videos directory where all outputs will be stored (e.g., /path/to/Videos)')
    parser.add_argument('--model', type=str, default='gpt4o_mini',
                        choices=['gpt4o_mini', 'gpt4o'],
                        help='Model to use for caption and Q&A generation')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing of all steps, even if outputs already exist')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    
    # Process the video
    results = process_single_video(args.video_path, args.base_videos_dir, args.model, args.force_reprocess)
    
    # Save results
    results_file = os.path.join(args.base_videos_dir, f"{results['video_name']}_processing_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    return results

if __name__ == "__main__":
    main()