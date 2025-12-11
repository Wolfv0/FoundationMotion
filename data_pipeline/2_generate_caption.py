import os, json, glob, argparse, av, sys, random
import cv2
from moviepy import *
import time
import base64
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from openai import OpenAI


device = "cuda" if torch.cuda.is_available() else "cpu"

    
def extract_index(data, extract_ls):
    """
    Recursively traverse a (possibly nested) dictionary. If a value is a list,
    extract its i-th element. If the value is a dictionary, process it recursively.
    Otherwise, leave the value unchanged.
    """
    if isinstance(data, dict):
        return {k: extract_index(v, extract_ls) for k, v in data.items()}
    elif isinstance(data, list):
        try:
            return [data[i] for i in extract_ls] #data[extract_ls]
        except IndexError:
            raise IndexError(f"Index {i} out of range for list: {data}")
    else:
        return data
    
def chunk_into_n(lst, n):
    """Divide list into n chunks as evenly as possible"""
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    
    chunks = []
    start = 0
    for i in range(n):
        # Add one extra item to the first 'remainder' chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    
    return chunks

def is_video_skipped_for_camera_motion(video_name, video_dir):
    """Check if a video was skipped due to camera motion detection"""
    camera_motion_marker = os.path.join(video_dir, f"{video_name}_camera_motion_skipped")
    return os.path.exists(camera_motion_marker)
    
def process_video(video_path, motion_info=None, image_caption_info=None, seconds_per_frame=None):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    extracted_ls = []
    while curr_frame < total_frames - 1:
        extracted_ls.append(curr_frame)
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # extracted_ls = [0]
    extracted_motion_info = extract_index(motion_info, extracted_ls)
    
    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    # clip = VideoFileClip(video_path)
    # clip.audio.write_audiofile(audio_path, bitrate="32k")
    # clip.audio.close()
    # clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    if image_caption_info is None:
        return base64Frames, extracted_motion_info, audio_path
    else:
        # extract motion info
        
        extracted_image_caption_info = [image_caption_info[i] for i in extracted_ls]
        
        return base64Frames, extracted_motion_info, extracted_image_caption_info, audio_path


def gpt4o_video_caption(model, client, video_path, motion_path=None):
    
    # prep motion
    with open(motion_path, 'r') as f:
        motion_info = json.load(f)
        
    # prep video
    base64Frames, extracted_motion_info, audio_path = process_video(video_path, motion_info, None, seconds_per_frame=0.5)
    
    # Get prompt path from environment variable, fallback to relative path
    prompt_path = os.environ.get('PROMPT_CAPTION_PATH', 'prompts/video_caption_1K_general.prompt')
    print(f'Prompt used: {prompt_path}')
    with open(prompt_path, 'r') as f:
        prompt = f.read()
        
    prompt = prompt.format(motion_info=extracted_motion_info)
    
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": prompt
                    },
        {"role": "user", "content": [
            "These are the frames from the video.",
            *map(lambda x: {"type": "image_url", 
                            "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
            ],
        }
        ],
        temperature=0,
    )
    response = response.choices[0].message.content
    print(f'sucessfully generated caption')
    return response



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt4o_mini", type=str, required=True)
    parser.add_argument("--seed", default=0, type=int, required=False)
    parser.add_argument("--video_dir", default="/data/vision/torralba/selfmanaged/isola/u/yulu/cmar/Videos/video_general_obj_det_finished", 
                        type=str, help="Directory containing video and motion files")
    parser.add_argument('--video_names', type=str, nargs='+', default=[], help='process only these specific videos (video names without extension)')
    parser.add_argument('--chunk_idx', type=int, default=None,
                        help='Chunk index to process (for parallel processing)')
    parser.add_argument('--chunk_num', type=int, default=None,
                        help='Total number of chunks (for parallel processing)')
    args = parser.parse_args()
    
    video_folder = args.video_dir
    motion_folder = video_folder
    
    # Check if video directory exists
    if not os.path.exists(video_folder):
        raise ValueError(f"Video directory does not exist: {video_folder}")
    
    # Check if processing specific videos
    if args.video_names:
        # Filter to only the specified videos
        video_ls = []
        missing_videos = []
        missing_motion_files = []
        
        for video_name in args.video_names:
            video_path = os.path.join(video_folder, f"{video_name}.mp4")
            motion_path = os.path.join(motion_folder, f"{video_name}.json")
            
            if os.path.exists(video_path) and os.path.exists(motion_path):
                video_ls.append(video_path)
            else:
                if not os.path.exists(video_path):
                    missing_videos.append(video_name)
                if not os.path.exists(motion_path):
                    missing_motion_files.append(video_name)
        
        if missing_videos:
            print(f'WARNING: Videos not found: {missing_videos}')
        if missing_motion_files:
            print(f'WARNING: Motion files not found: {missing_motion_files}')
        
        if video_ls:
            print(f'Processing SPECIFIC videos: {len(video_ls)} videos from provided list')
            print(f'Videos to process: {[os.path.splitext(os.path.basename(v))[0] for v in video_ls]}')
        else:
            print(f'ERROR: None of the specified videos found with motion files')
            exit(1)
    else:
        # NEW APPROACH: Use atomic snapshot for absolute consistency
        snapshot_file = os.path.join(os.path.dirname(video_folder), ".atomic_snapshot.json")
        
        if os.path.exists(snapshot_file):
            print("Using atomic snapshot for consistent chunking")
            
            import json
            try:
                with open(snapshot_file, 'r') as f:
                    snapshot_data = json.load(f)
                
                video_names = snapshot_data.get('videos', [])
                print(f"Snapshot contains {len(video_names)} videos: {video_names}")
                
                # Convert to full paths and verify they exist
                video_ls = []
                for video_name in video_names:
                    video_path = os.path.join(video_folder, f"{video_name}.mp4")
                    if os.path.exists(video_path):
                        video_ls.append(video_path)
                    else:
                        print(f"Warning: Video from snapshot not found: {video_path}")
                
                print(f"Found {len(video_ls)} videos from atomic snapshot")
                
            except Exception as e:
                print(f"Error reading snapshot: {e}, falling back to dynamic scanning")
                # Fall through to dynamic scanning
                video_ls = None
        else:
            print("No atomic snapshot found, using dynamic scanning")
            video_ls = None
    
        # Fallback to dynamic scanning if snapshot failed
        if video_ls is None or len(video_ls) == 0:
            print("Using dynamic file scanning as fallback")
            video_ls = glob.glob(os.path.join(video_folder, "*.mp4"))
            if not video_ls:
                raise ValueError(f"No mp4 files found in directory: {video_folder}")
            print(f"Found {len(video_ls)} video files")
            
            # Get all motion files
            motion_files = glob.glob(os.path.join(motion_folder, "*.json"))
            if not motion_files:
                raise ValueError(f"No json files found in directory: {motion_folder}")
            print(f"Found {len(motion_files)} motion files")
            
            # Filter videos to only include those that have corresponding motion files
            motion_names = set(os.path.splitext(os.path.basename(m))[0] for m in motion_files)
            
            valid_videos = []
            skipped_camera_motion = []
            for video_path in video_ls:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                # Check if video was skipped due to camera motion
                if is_video_skipped_for_camera_motion(video_name, video_folder):
                    skipped_camera_motion.append(video_name)
                    continue
                if video_name in motion_names:
                    valid_videos.append(video_path)
            
            # CRITICAL: Sort to ensure consistent ordering across all chunks
            valid_videos.sort()
            video_ls = valid_videos
            print(f"Found {len(video_ls)} videos with corresponding motion files")
            if skipped_camera_motion:
                print(f"Excluded {len(skipped_camera_motion)} videos skipped due to camera motion")
    
    # ROBUST CHUNKING: Create stable chunk assignments
    if args.chunk_idx is not None and args.chunk_num is not None:
        if len(video_ls) == 0:
            print(f"No videos to process in chunk {args.chunk_idx}")
            exit(0)
        
        # Create a deterministic and stable chunk assignment file - SHARED for both caption and QA
        chunk_assignment_dir = os.path.join(os.path.dirname(video_folder), ".chunk_assignments")
        os.makedirs(chunk_assignment_dir, exist_ok=True)
        chunk_assignment_file = os.path.join(chunk_assignment_dir, f"video_chunks_{args.chunk_num}.txt")
        
        # Generate chunk assignments if they don't exist
        if not os.path.exists(chunk_assignment_file):
            print(f"Creating stable chunk assignments for {args.chunk_num} chunks")
            
            # Sort video list to ensure deterministic ordering
            sorted_videos = sorted([os.path.basename(v)[:-4] for v in video_ls])  # Remove .mp4 extension
            
            # Pre-allocate videos to chunks
            chunk_assignments = [[] for _ in range(args.chunk_num)]
            for i, video_name in enumerate(sorted_videos):
                chunk_assignments[i % args.chunk_num].append(video_name)
            
            # Write assignments to file
            with open(chunk_assignment_file, 'w') as f:
                for chunk_id, videos in enumerate(chunk_assignments):
                    f.write(f"CHUNK_{chunk_id}:" + ",".join(videos) + "\n")
            
            print(f"Chunk assignments created: {len(sorted_videos)} videos distributed across {args.chunk_num} chunks")
        
        # Read chunk assignments
        with open(chunk_assignment_file, 'r') as f:
            for line in f:
                if line.startswith(f"CHUNK_{args.chunk_idx}:"):
                    assigned_videos = line.strip().split(":", 1)[1].split(",")
                    if assigned_videos == [""]:  # Empty chunk
                        assigned_videos = []
                    break
            else:
                assigned_videos = []
        
        # Filter video_ls to only include assigned videos that actually exist
        chunk_videos = []
        for video_path in video_ls:
            video_name = os.path.basename(video_path)[:-4]  # Remove .mp4 extension
            if video_name in assigned_videos:
                # Verify the video file actually exists
                if os.path.exists(video_path):
                    chunk_videos.append(video_path)
                else:
                    print(f"Warning: Assigned video not found: {video_path}")
        
        video_ls = chunk_videos
        print(f"Processing chunk {args.chunk_idx}/{args.chunk_num-1} with {len(video_ls)} videos (stable pre-allocated assignment)")
        print(f"Assigned videos: {[os.path.basename(v)[:-4] for v in video_ls]}")
        
        if len(video_ls) == 0:
            print(f"No valid videos found for chunk {args.chunk_idx}")
            exit(0)
    else:
        print(f"Processing all {len(video_ls)} videos")
    
    # OpenAI client setup
    traver_api_key = "your_openai_api_key"
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", traver_api_key))
    save_dir = os.path.join(os.path.dirname(video_folder), f'videos_captions_{args.model}')

    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Output directory: {save_dir}")
    except Exception as e:
        raise ValueError(f"Cannot create output directory {save_dir}: {str(e)}")
    
    
    # Track progress for resume functionality
    total_videos = len(video_ls)
    processed_count = 0
    skipped_count = 0
    generated_count = 0
    
    print(f"Caption generation progress tracking:")
    print(f"Total videos to process: {total_videos}")
    
    for v_idx, video_path in tqdm(enumerate(video_ls)):
        video_name = video_path.split('/')[-1][:-4]
        
        save_path  = os.path.join(save_dir, video_name+'.json')
        
        # motion
        motion_path = os.path.join(motion_folder, f"{video_name}.json")
        if os.path.exists(video_path) and os.path.exists(motion_path):
            GPT_4o_model = "gpt-4o-mini" if args.model == "gpt4o_mini" else "gpt-4o"
            caption = gpt4o_video_caption(GPT_4o_model, client, video_path, motion_path)
                
          
            res = {f'{args.model}_res': caption}
            with open(save_path, 'w') as f:
                json.dump(res, f, indent=4)
            
            generated_count += 1
            print(f'✓ Generated caption for: {video_name}')
            
        else:
            print(f'Files not found - Video: {video_path}, Motion: {motion_path}')
    
    # Final progress summary
    print(f"\n=== CAPTION GENERATION SUMMARY ===")
    print(f"Total videos: {total_videos}")
    print(f"Already completed (skipped): {skipped_count}")
    print(f"Newly generated: {generated_count}")
    print(f"Total completed: {skipped_count + generated_count}")
    
    if generated_count > 0:
        print(f" Successfully generated {generated_count} new captions")
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} videos (already had captions)")
    
    completion_rate = ((skipped_count + generated_count) / total_videos * 100) if total_videos > 0 else 0
    print(f"Overall completion rate: {completion_rate:.1f}%")