"""This is the vp baseline: using qwen 2 caption videos, then to get pickobj name and slot name, and them use grounded-sam2, to get masks in robot view.
"""
import os, json, glob, argparse, av, sys
import cv2
from moviepy import *
import time
import base64
import random

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import ast
from pydantic import BaseModel


class OneQA(BaseModel):
    question: str
    A: str
    B: str
    C: str
    D: str

class QAGenerating(BaseModel):
    questions: list[OneQA]


def randomize_qa_answers(qa_data):
    for qa in qa_data['questions']:
        options = [qa['A'], qa['B'], qa['C'], qa['D']]
        correct_answer_text = qa['A']
        
        random.shuffle(options)
        
        qa['A'] = options[0]
        qa['B'] = options[1] 
        qa['C'] = options[2]
        qa['D'] = options[3]
        
        for i, option in enumerate(options):
            if option == correct_answer_text:
                qa['correct_answer'] = ['A', 'B', 'C', 'D'][i]
                break
    
    return qa_data


from openai import OpenAI

device = "cuda" if torch.cuda.is_available() else "cpu"

    

def generate_gpt4o_QA(model, client, caption_path, prompt_path):
    
    # load caption
    with open(caption_path, 'r') as f:
        caption_info = json.load(f)
        
    # load prompt
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    prompt = prompt.format(video_caption=caption_info)
    
    response = client.chat.completions.create(
    model=model,
    messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    response = response.choices[0].message.content
    print(f'sucessfully generated QA')
    response = json.loads(response)
    
    response = randomize_qa_answers(response)
    
    return response

def process_video(video_path, seconds_per_frame=None):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

  
    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")

    return base64Frames
 
    
def generate_gpt4o_QA_video(model, client, caption_path, video_path, prompt_path):
    
    # load caption
    with open(caption_path, 'r') as f:
        caption_info = json.load(f)
        
    # prep video
    base64Frames  = process_video(video_path, seconds_per_frame=1)
    
    # load prompt
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    # breakpoint()
        
    prompt = prompt.format(video_caption=caption_info)
    
    # response = client.chat.completions.create(
    response = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            "These are the frames from the video.",
            *map(lambda x: {"type": "image_url", 
                            "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
            ],
        }
        ],
        temperature=0,
        response_format=QAGenerating
    )
    response = response.choices[0].message.content
    print(f'gpt_4o QA output = \n{response}')
    response = json.loads(f"""{response}""")
    
    response = randomize_qa_answers(response)
    
    return response

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




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt4o_mini", type=str, required=False)
    parser.add_argument("--seed", default=None, type=int, required=False)
    parser.add_argument("--video_dir", default="/data/vision/torralba/selfmanaged/isola/u/yulu/cmar/Videos/videos_bodyhands", 
                        type=str, help="Directory containing video files")
    parser.add_argument("--caption_dir", default="/data/vision/torralba/selfmanaged/isola/u/yulu/cmar/videos_captions_gpt4o_mini", 
                        type=str, help="Directory containing caption files")
    parser.add_argument("--prompt_path", default=os.environ.get('PROMPT_QA_PATH', "/data/vision/torralba/selfmanaged/isola/u/yulu/cmar/Data-Test/vla_motion/prompts/caption_QA.prompt"), 
                        type=str, help="Path to the prompt file")
    parser.add_argument('--video_names', type=str, nargs='+', default=[], help='process only these specific videos (video names without extension)')
    parser.add_argument('--chunk_idx', type=int, default=None,
                        help='Chunk index to process (for parallel processing)')
    parser.add_argument('--chunk_num', type=int, default=None,
                        help='Total number of chunks (for parallel processing)')
    parser.add_argument('--caption_list_file', type=str, default=None,
                        help='Path to a file containing a list of caption files to process')
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    else:
        random.seed()
        print("Using random seed")
    
    # Check if paths exist
    if not os.path.exists(args.video_dir):
        raise ValueError(f"Video directory does not exist: {args.video_dir}")
    
    if not os.path.exists(args.caption_dir):
        raise ValueError(f"Caption directory does not exist: {args.caption_dir}")
    
    if not os.path.exists(args.prompt_path):
        raise ValueError(f"Prompt file does not exist: {args.prompt_path}")
    
    # Setup OpenAI client
    GPT_4o_model = "gpt-4o-mini" if args.model == "gpt4o_mini" else "gpt-4o"
    traver_api_key = "your_openai_api_key"
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", traver_api_key))
    
    # Setup output directory
    save_dir = os.path.join(os.path.dirname(args.video_dir), f'videos_QAs_{args.model}')
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Output directory: {save_dir}")
    except Exception as e:
        raise ValueError(f"Cannot create output directory {save_dir}: {str(e)}")
    
    # Check if processing specific videos
    if args.video_names:
        # Filter to only the specified videos
        video_ls = []
        missing_videos = []
        missing_captions = []
        
        for video_name in args.video_names:
            video_path = os.path.join(args.video_dir, f"{video_name}.mp4")
            caption_path = os.path.join(args.caption_dir, f"{video_name}.json")
            
            if os.path.exists(video_path) and os.path.exists(caption_path):
                video_ls.append(video_path)
            else:
                if not os.path.exists(video_path):
                    missing_videos.append(video_name)
                if not os.path.exists(caption_path):
                    missing_captions.append(video_name)
        
        if missing_videos:
            print(f'WARNING: Videos not found: {missing_videos}')
        if missing_captions:
            print(f'WARNING: Caption files not found: {missing_captions}')
        
        if video_ls:
            print(f'Processing SPECIFIC videos: {len(video_ls)} videos from provided list')
            print(f'Videos to process: {[os.path.splitext(os.path.basename(v))[0] for v in video_ls]}')
        else:
            print(f'ERROR: None of the specified videos found with caption files')
            exit(1)
    else:
        # NEW APPROACH: Use atomic snapshot but filter by available captions
        snapshot_file = os.path.join(os.path.dirname(args.video_dir), ".atomic_snapshot.json")
        
        # Get caption files - either from list file or by scanning directory
        if args.caption_list_file and os.path.exists(args.caption_list_file):
            print(f"Using caption list file: {args.caption_list_file}")
            try:
                with open(args.caption_list_file, 'r') as f:
                    caption_files = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(caption_files)} caption files from list")
            except Exception as e:
                print(f"Error reading caption list file: {e}")
                print("Falling back to directory scan")
                caption_files = glob.glob(os.path.join(args.caption_dir, "*.json"))
        else:
            # Get all caption files by scanning directory
            caption_files = glob.glob(os.path.join(args.caption_dir, "*.json"))
        
        if not caption_files:
            print(f"No caption files found in directory: {args.caption_dir}")
            video_ls = []  # No captions means no videos to process
        else:
            print(f"Found {len(caption_files)} caption files")
            caption_names = set(os.path.splitext(os.path.basename(c))[0] for c in caption_files)
            
            # Try to use atomic snapshot if available
            if os.path.exists(snapshot_file):
                print("Using atomic snapshot filtered by available captions")
                
                import json
                try:
                    with open(snapshot_file, 'r') as f:
                        snapshot_data = json.load(f)
                    
                    snapshot_videos = snapshot_data.get('videos', [])
                    print(f"Snapshot contains {len(snapshot_videos)} videos")
                    
                    # Filter snapshot videos by those that have captions
                    video_ls = []
                    skipped_camera_motion = []
                    for video_name in snapshot_videos:
                        # Check if video was skipped due to camera motion
                        if is_video_skipped_for_camera_motion(video_name, args.video_dir):
                            skipped_camera_motion.append(video_name)
                            continue
                        if video_name in caption_names:
                            video_path = os.path.join(args.video_dir, f"{video_name}.mp4")
                            if os.path.exists(video_path):
                                video_ls.append(video_path)
                            else:
                                print(f"Warning: Video from snapshot not found: {video_path}")
                    
                    print(f"Found {len(video_ls)} videos from snapshot that have captions")
                    if skipped_camera_motion:
                        print(f"Excluded {len(skipped_camera_motion)} videos skipped due to camera motion")
                    
                except Exception as e:
                    print(f"Error reading snapshot: {e}, falling back to dynamic scanning")
                    video_ls = None
            else:
                print("No atomic snapshot found, using dynamic scanning")
                video_ls = None
            
            # Fallback to dynamic scanning if snapshot failed
            if video_ls is None:
                print("Using dynamic scanning based on available caption files")
                
                # Build video list based on caption files that exist
                video_ls = []
                skipped_camera_motion = []
                for caption_name in caption_names:
                    # Check if video was skipped due to camera motion
                    if is_video_skipped_for_camera_motion(caption_name, args.video_dir):
                        skipped_camera_motion.append(caption_name)
                        continue
                    video_path = os.path.join(args.video_dir, f"{caption_name}.mp4")
                    if os.path.exists(video_path):
                        video_ls.append(video_path)
                    else:
                        print(f"Warning: Video file not found for caption: {video_path}")
                
                if skipped_camera_motion:
                    print(f"Excluded {len(skipped_camera_motion)} videos skipped due to camera motion")
            
            # CRITICAL: Sort to ensure consistent ordering across all chunks
            video_ls.sort()
            print(f"Final video list: {len(video_ls)} videos with corresponding caption files")
    
    # ROBUST CHUNKING: Create stable chunk assignments
    if args.chunk_idx is not None and args.chunk_num is not None:
        if len(video_ls) == 0:
            print(f"No videos to process in chunk {args.chunk_idx}")
            exit(0)
        
        # Use the SAME chunk assignment file as caption generation
        chunk_assignment_dir = os.path.join(os.path.dirname(args.video_dir), ".chunk_assignments")
        chunk_assignment_file = os.path.join(chunk_assignment_dir, f"video_chunks_{args.chunk_num}.txt")
        
        # QA generation should NEVER create chunk assignments - only use existing ones from caption stage
        if not os.path.exists(chunk_assignment_file):
            print(f"ERROR: Chunk assignment file not found: {chunk_assignment_file}")
            print("This should have been created during caption generation stage.")
            print("Cannot proceed with QA generation without consistent chunk assignments.")
            exit(1)
        
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
        
        # Filter video_ls to only include assigned videos
        chunk_videos = []
        for video_path in video_ls:
            video_name = os.path.basename(video_path)[:-4]  # Remove .mp4 extension
            if video_name in assigned_videos:
                chunk_videos.append(video_path)
        
        video_ls = chunk_videos
        print(f"Processing chunk {args.chunk_idx}/{args.chunk_num-1} with {len(video_ls)} videos (stable pre-allocated assignment)")
        print(f"Assigned videos: {[os.path.basename(v)[:-4] for v in video_ls]}")
    else:
        print(f"Processing all {len(video_ls)} videos")
    
    # Track progress for resume functionality
    total_videos = len(video_ls)
    processed_count = 0
    skipped_count = 0
    generated_count = 0
    missing_caption_count = 0
    
    print(f"QA generation progress tracking:")
    print(f"Total videos to process: {total_videos}")
    
    for v_idx, video_path in tqdm(enumerate(video_ls)):
        video_name = video_path.split('/')[-1][:-4]
        caption_path = os.path.join(args.caption_dir, video_name+'.json')
        save_path = os.path.join(save_dir, video_name+'.json')
        
        # Check if caption file exists
        if not os.path.exists(caption_path):
            print(f'Caption file not found: {caption_path}')
            missing_caption_count += 1
            continue
        
        # Generate QA
        caption = generate_gpt4o_QA_video(GPT_4o_model, client, caption_path, video_path, args.prompt_path)
    
        # Save results
        res = {
            f'{args.model}_res': caption,
            'metadata': {
                'randomized_answers': True,
                'correct_answer_field': 'correct_answer',
                'description': 'Answers are randomized. Check correct_answer field for each question.'
            }
        }
        with open(save_path, 'w') as f:
            json.dump(res, f, indent=4)
        
        generated_count += 1
        print(f'✓ Generated QA for: {video_name}')
    
    # Final progress summary
    print(f"\n=== QA GENERATION SUMMARY ===")
    print(f"Total videos: {total_videos}")
    print(f"Already completed (skipped): {skipped_count}")
    print(f"Newly generated: {generated_count}")
    print(f"Missing captions: {missing_caption_count}")
    print(f"Total completed: {skipped_count + generated_count}")
    
    if generated_count > 0:
        print(f"✅ Successfully generated {generated_count} new QA files")
    
    if skipped_count > 0:
        print(f"⏭️  Skipped {skipped_count} videos (already had QA)")
        
    if missing_caption_count > 0:
        print(f"⚠️  {missing_caption_count} videos missing caption files")
    
    completion_rate = ((skipped_count + generated_count) / total_videos * 100) if total_videos > 0 else 0
    print(f"Overall completion rate: {completion_rate:.1f}%")