import os
import glob
import random
import subprocess
import json
import argparse
from tqdm import tqdm

def get_video_duration(video_path):
    """Get the duration of a video using ffprobe"""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "json", 
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def crop_video(video_path, output_path, start_time, duration):
    """Crop a segment from a video"""
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", video_path,
        "-ss", str(start_time),
        "-t", str(duration),
        # Re-encode for compatibility to ensure valid keyframes and timestamps
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd)

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

def process_videos(input_dir, output_dir, chunk_idx=None, chunk_num=None):
    """Process all videos in the input directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files (assuming mp4, but can be extended)
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    video_files.extend(glob.glob(os.path.join(input_dir, "*.avi")))
    video_files.extend(glob.glob(os.path.join(input_dir, "*.mov")))
    
    # Sort for consistent chunking across runs
    video_files.sort()
    
    # Handle chunking
    if chunk_idx is not None and chunk_num is not None:
        video_chunks = chunk_into_n(video_files, chunk_num)
        if chunk_idx >= len(video_chunks):
            print(f"Chunk {chunk_idx} is out of range (total chunks: {len(video_chunks)})")
            return
        video_files = video_chunks[chunk_idx]
        print(f"Processing chunk {chunk_idx}/{chunk_num-1} with {len(video_files)} videos")
    else:
        print(f"Found {len(video_files)} videos to process")
    
    for video_path in tqdm(video_files):
        # Get video filename
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, video_name)
        
        # Get video duration
        duration = get_video_duration(video_path)
        
        if duration <= 5.0:
            # If video is shorter than 5s, copy it directly
            print(f"Video {video_name} is {duration:.2f}s (≤5s), copying directly")
            cmd = ["cp", video_path, output_path]
            subprocess.run(cmd)
        else:
            # Choose a random duration between 5-10s
            crop_duration = random.uniform(5.0, min(10.0, duration))
            
            # Calculate start time (try to position in the middle)
            max_start = duration - crop_duration
            middle_point = duration / 2 - crop_duration / 2
            
            # Add some randomness but keep it near the middle
            # Start time will be within 20% of the video length from the middle position
            variation = min(duration * 0.2, max_start / 2)
            start_time = max(0, min(max_start, middle_point + random.uniform(-variation, variation)))
            
            print(f"Video {video_name} is {duration:.2f}s, cropping {crop_duration:.2f}s from position {start_time:.2f}s")
            crop_video(video_path, output_path, start_time, crop_duration)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crop videos to 5-10 second segments')
    parser.add_argument('--in_dir', type=str, default='/path/to/Videos',
                        help='Directory containing input videos')
    parser.add_argument('--out_dir', type=str, default='/path/to/Videos_crop',
                        help='Directory to save cropped videos')
    parser.add_argument('--chunk_idx', type=int, default=None,
                        help='Chunk index to process (for parallel processing)')
    parser.add_argument('--chunk_num', type=int, default=None,
                        help='Total number of chunks (for parallel processing)')
    args = parser.parse_args()
    
    # Process videos
    process_videos(args.in_dir, args.out_dir, args.chunk_idx, args.chunk_num)
    print(f"Processing complete. Cropped videos saved to {args.out_dir}")
