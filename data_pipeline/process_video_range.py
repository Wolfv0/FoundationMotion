"""
Process a specific index range of videos in a directory.

This utility lists videos in a deterministic order, slices by [start_idx:end_idx),
and processes each using optimized batch processing to avoid reloading models.

Usage:
    python process_video_range.py \
        --video_dir /path/to/videos \
        --base_videos_dir /path/to/Videos \
        --start_idx 0 --end_idx 10 \
        --model gpt4o_mini
"""

import os
import sys
import glob
import json
import argparse
import subprocess
import shutil
from typing import List, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import individual steps from process_single_video (excluding object detection)
from process_single_video import (
    setup_environment, 
    step_preprocess, 
    step_0_decode, 
    step_2_hand_caption, 
    step_3_generate_qa,
    check_video_processing_status
)


def list_videos(video_dir: str, video_extensions: List[str]) -> List[str]:
    """Return sorted list of video file paths matching extensions in the directory."""
    files: List[str] = []
    for ext in video_extensions:
        pattern = os.path.join(video_dir, ext)
        files.extend(glob.glob(pattern))
    files.sort()
    return files

def is_video_skipped_for_camera_motion(video_name: str, base_videos_dir: str) -> bool:
    """Check if a video was skipped due to camera motion detection"""
    # Check in the object detection output directory
    obj_det_dir = os.path.join(base_videos_dir, "video_general_obj_det_finished")
    camera_motion_marker = os.path.join(obj_det_dir, f"{video_name}_camera_motion_skipped")

    return os.path.isdir(camera_motion_marker)


def resolve_index_range(total: int, start_idx: int, end_idx: int) -> Tuple[int, int]:
    """Clamp and normalize the requested [start_idx, end_idx) range against total size."""
    if start_idx < 0:
        start_idx = 0
    if end_idx < 0:
        end_idx = 0
    if end_idx > total:
        end_idx = total
    if start_idx > total:
        start_idx = total
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    return start_idx, end_idx


def process_single_video_phase1(video_path: str, base_videos_dir: str, model: str = "gpt4o_mini", force_reprocess: bool = False) -> dict:
    """
    Process a single video through preprocessing steps (preprocess + decode only).
    Object detection, captions, and QA will be done later in batch.
    """
    print(f"Phase 1 - Processing video: {video_path}")
    
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
        
        results["status"] = "success"
        print(f"✓ Phase 1 completed for video: {video_name}")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        print(f"✗ Phase 1 failed for video {video_name}: {e}")
        raise
    
    return results


def process_single_video_phase3(video_path: str, base_videos_dir: str, model: str = "gpt4o_mini") -> dict:
    """
    Process a single video through final steps (captions + QA) after object detection is complete.
    """
    print(f"Phase 3 - Processing video: {video_path}")
    
    # Get video name for tracking
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    results = {
        "video_name": video_name,
        "original_video": video_path,
        "base_videos_dir": base_videos_dir
    }
    
    # Check if captions and QA are already completed
    status = check_video_processing_status(video_path, base_videos_dir, model)
    if status["captions"] and status["qa"]:
        results["status"] = "already_completed"
        results["skip_reason"] = "captions_and_qa_exist"
        print(f"⏭️  Phase 3 - Skipping {video_name} (captions and QA already exist)")
        return results
    
    # Check if video was skipped due to camera motion (after object detection)
    if is_video_skipped_for_camera_motion(video_name, base_videos_dir):
        results["status"] = "skipped_camera_motion"
        results["skip_reason"] = "camera_motion"
        print(f"⏭️  Phase 3 - Skipping {video_name} (camera motion detected)")
        return results
    
    # Also check if motion file exists (fallback check for skipped videos)
    obj_det_dir = os.path.join(base_videos_dir, "video_general_obj_det_finished")
    motion_file = os.path.join(obj_det_dir, f"{video_name}.json")
    if not os.path.exists(motion_file):
        results["status"] = "skipped_no_motion_file"
        results["skip_reason"] = "no_motion_file"
        print(f"⏭️  Phase 3 - Skipping {video_name} (no motion file found)")
        return results
    
    try:
        # Object detection directory should exist after batch processing
        obj_det_dir = os.path.join(base_videos_dir, "video_general_obj_det_finished")
        
        # Extract video name from video path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Step 2: Hand captions -> videos_captions_{model}/
        captions_dir = step_2_hand_caption(obj_det_dir, base_videos_dir, model, [video_name])
        results["captions"] = captions_dir
        
        # Step 3: Q&A generation -> videos_QAs_{model}/
        qa_dir = step_3_generate_qa(obj_det_dir, captions_dir, base_videos_dir, model, [video_name])
        results["qa"] = qa_dir
        
        results["status"] = "success"
        print(f"✓ Phase 3 completed for video: {video_name}")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        print(f"✗ Phase 3 failed for video {video_name}: {e}")
        raise
    
    return results


def run_batch_object_detection(base_videos_dir: str, video_names: List[str]) -> bool:
    """
    Run object detection for multiple videos in a single batch to avoid model reloading.
    """
    print(f"=== BATCH OBJECT DETECTION: Processing {len(video_names)} videos ===")
    
    # Prepare video directory structure
    videos_dir = os.path.join(base_videos_dir, "Videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Copy videos to Videos directory if needed
    for video_name in video_names:
        videos_crop_dir = os.path.join(base_videos_dir, "Videos_crop")
        source_video = os.path.join(videos_crop_dir, f"{video_name}.mp4")
        target_video = os.path.join(videos_dir, f"{video_name}.mp4")
        
        if os.path.exists(source_video) and not os.path.exists(target_video):
            shutil.copy2(source_video, target_video)
    
    print(f"Videos to process: {video_names[:5]}{'...' if len(video_names) > 5 else ''}")
    
    # Run batch object detection using 1_general_obj_det_v4.py with specific video names
    # Pass video names directly as command line arguments
    video_names_str = ' '.join(video_names)
    cmd = f"""
    accelerate launch \
        --num_processes 1 \
        --num_machines 1 \
        --machine_rank 0 \
        --mixed_precision fp16 \
        1_general_obj_det_v4.py \
        --video_dir={base_videos_dir} \
        --enable_camera_motion_detection \
        --video_names {video_names_str}
    """
    
    print(f"Running batch object detection command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"Batch object detection failed with return code {result.returncode}")
        return False
    
    print(f"✓ Batch object detection completed successfully")
    return True


def process_videos_in_range(
    video_dir: str,
    base_videos_dir: str,
    start_idx: int,
    end_idx: int,
    model: str = "gpt4o_mini",
    video_extensions: List[str] = None,
    force_reprocess: bool = False,
) -> dict:
    """
    Process videos in [start_idx, end_idx) from the given directory.
    Uses optimized 3-phase batch processing to avoid model reloading.

    Returns a summary dict with selected files and per-video results.
    """
    if video_extensions is None:
        video_extensions = ["*.mp4", "*.avi", "*.mov"]

    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    os.makedirs(base_videos_dir, exist_ok=True)

    all_videos = list_videos(video_dir, video_extensions)
    total = len(all_videos)
    if total == 0:
        raise ValueError(
            f"No video files found in {video_dir} with extensions {video_extensions}"
        )

    s, e = resolve_index_range(total, start_idx, end_idx)
    selected = all_videos[s:e]

    print(f"Found {total} videos in: {video_dir}")
    print(f"Processing range [{s}:{e}) -> {len(selected)} videos")
    print(f"Using 3-phase processing to optimize model loading")

    # Phase 1: Process individual video steps (preprocess, decode only)
    phase1_results = []
    video_names_for_batch = []
    
    print(f"\n=== PHASE 1: Preprocessing and Decoding ===")
    for idx, video_path in enumerate(selected, start=s):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Phase 1 - Processing {idx}/{total - 1}: {os.path.basename(video_path)}")
        try:
            res = process_single_video_phase1(
                video_path=video_path,
                base_videos_dir=base_videos_dir,
                model=model,
                force_reprocess=force_reprocess,
            )
            phase1_results.append({"video": video_path, "status": "success", "detail": res})
            
            # Collect video names for batch object detection
            if res["status"] == "success":
                video_names_for_batch.append(res["video_name"])
                
        except Exception as exc:
            phase1_results.append({"video": video_path, "status": "failed", "error": str(exc)})

    # Phase 2: Batch object detection for all videos
    batch_success = True  # Default to True if no videos to process
    print(f"\n=== PHASE 2: Batch Object Detection ===")
    if video_names_for_batch:
        print(f"Running batch object detection for {len(video_names_for_batch)} videos")
        batch_success = run_batch_object_detection(base_videos_dir, video_names_for_batch)
        
        if not batch_success:
            print("⚠️  Batch object detection failed")
    else:
        print("No videos to process for object detection")

    # Phase 3: Process captions and QA for successfully processed videos
    final_results = []
    skipped_camera_motion = []
    print(f"\n=== PHASE 3: Captions and Q&A Generation ===")
    
    if batch_success and video_names_for_batch:
        for phase1_result in phase1_results:
            if phase1_result["status"] == "success":
                video_path = phase1_result["video"]
                print(f"Phase 3 - Processing {os.path.basename(video_path)}")
                try:
                    phase3_res = process_single_video_phase3(
                        video_path=video_path,
                        base_videos_dir=base_videos_dir,
                        model=model,
                    )
                    
                    # Check if video was skipped due to camera motion, no motion file, or already completed
                    if phase3_res["status"] in ["skipped_camera_motion", "skipped_no_motion_file", "already_completed"]:
                        if phase3_res["status"] != "already_completed":
                            skipped_camera_motion.append(phase3_res["video_name"])
                        final_results.append({"video": video_path, "status": phase3_res["status"], "detail": phase3_res})
                    else:
                        # Merge phase 1 and phase 3 results
                        combined_result = phase1_result["detail"].copy()
                        combined_result.update(phase3_res)
                        combined_result["status"] = "success"
                        
                        final_results.append({"video": video_path, "status": "success", "detail": combined_result})
                    
                except Exception as exc:
                    final_results.append({"video": video_path, "status": "failed", "error": str(exc)})
            else:
                # Keep failed phase 1 results
                final_results.append(phase1_result)
                
        if skipped_camera_motion:
            print(f"Excluded {len(skipped_camera_motion)} videos from caption/QA generation (camera motion or missing motion files)")
    else:
        # If batch object detection failed, keep phase 1 results but mark as incomplete
        for phase1_result in phase1_results:
            if phase1_result["status"] == "success":
                phase1_result["detail"]["status"] = "incomplete_object_detection_failed"
            final_results.append(phase1_result)

    summary = {
        "video_dir": video_dir,
        "base_videos_dir": base_videos_dir,
        "model": model,
        "range": {"start_idx": s, "end_idx": e},
        "total_available": total,
        "selected": selected,
        "results": final_results,
        "batch_object_detection": batch_success,
        "processing_phases": {
            "phase1_preprocessing": len([r for r in phase1_results if r["status"] == "success"]),
            "phase2_object_detection": batch_success,
            "phase3_captions_qa": len([r for r in final_results if r["status"] == "success"])
        }
    }

    # Persist a small summary next to base_videos_dir for bookkeeping
    out_file = os.path.join(
        base_videos_dir, f"range_{s}_{e}_processing_summary.json"
    )
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Process a specific index range of videos without SLURM"
    )
    parser.add_argument(
        "--video_dir", type=str, required=True, help="Directory containing videos"
    )
    parser.add_argument(
        "--base_videos_dir",
        type=str,
        required=True,
        help="Base Videos directory to store unified outputs (e.g., /path/to/Videos)",
    )
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument(
        "--model", type=str, default="gpt4o_mini", choices=["gpt4o_mini", "gpt4o"]
    )
    parser.add_argument(
        "--video_extensions",
        nargs="+",
        default=["*.mp4", "*.avi", "*.mov"],
        help="Video filename patterns to include",
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Force reprocessing even if outputs exist",
    )

    args = parser.parse_args()

    process_videos_in_range(
        video_dir=args.video_dir,
        base_videos_dir=args.base_videos_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        model=args.model,
        video_extensions=args.video_extensions,
        force_reprocess=args.force_reprocess,
    )


if __name__ == "__main__":
    main()

