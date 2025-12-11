import argparse
import os, glob, os, shutil, random
import sys
import subprocess
import json
from shutil import which
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

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

def _resolve_ffmpeg_binary() -> str:
    # 1) Respect env override
    env_bin = os.environ.get('FFMPEG')
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin
    # 2) Use system PATH
    sys_bin = which('ffmpeg')
    if sys_bin:
        return sys_bin
    # 3) Try alongside the current Python executable (conda env bin)
    py_dir = os.path.dirname(sys.executable)
    candidate = os.path.join(py_dir, 'ffmpeg')
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    # 4) Try known conda env path
    known_candidate = '/data/vision/torralba/selfmanaged/isola/u/yulu/miniconda3/envs/fm/bin/ffmpeg'
    if os.path.isfile(known_candidate) and os.access(known_candidate, os.X_OK):
        return known_candidate
    return 'ffmpeg'  # fallback; will likely fail with 'not found'

def has_video_stream(video_path):
    """Check if the video file contains a video stream."""
    ffmpeg_bin = _resolve_ffmpeg_binary()
    cmd = [ffmpeg_bin, "-i", video_path, "-hide_banner"]
    
    # Run ffmpeg with stderr redirected to stdout
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = process.communicate()[0]
    
    # Check if there's a video stream in the output
    return "Stream" in output and ("Video:" in output or "video:" in output)

def decode_video(video_path, save_dir, video_name=''):
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if the file has a video stream
    if not has_video_stream(video_path):
        print(f"Warning: {video_path} does not contain a video stream. Skipping...")
        # Create a marker file to indicate this is an audio-only file
        with open(os.path.join(save_dir, 'audio_only.txt'), 'w') as f:
            f.write(f"This file ({video_path}) contains only audio, no video stream to extract frames from.")
        return
    
    # Use more robust ffmpeg parameters to handle videos with timestamp issues
    ffmpeg_bin = _resolve_ffmpeg_binary()
    cmd = f"{ffmpeg_bin} -loglevel error -fflags +genpts -i {video_path} -r 30 -f image2 -qscale:v 1 '{save_dir}/%10d.jpg'"
    result = os.system(cmd)
    frame_count = len(os.listdir(save_dir))
    print(f"{video_path}, #frames={frame_count}")
    
    # If no frames extracted, try alternative approach
    if frame_count == 0:
        print(f"Warning: No frames extracted with standard method, trying alternative approach...")
        # Clear directory first
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
        # Try with video filter to force frame extraction
        cmd_alt = f"{ffmpeg_bin} -loglevel error -fflags +genpts -i {video_path} -vf 'fps=30' -qscale:v 1 '{save_dir}/%10d.jpg'"
        result_alt = os.system(cmd_alt)
        frame_count_alt = len(os.listdir(save_dir))
        print(f"{video_path}, #frames={frame_count_alt} (alternative method)")
        
        # If still no frames, mark as problematic
        if frame_count_alt == 0:
            with open(os.path.join(save_dir, 'extraction_failed.txt'), 'w') as f:
                f.write(f"Failed to extract frames from {video_path} using both standard and alternative methods.")
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Decode videos to frames')
    parser.add_argument('--clip_dir', type=str, default='', help='Folder with input video folder')
    parser.add_argument('--chunk_idx', type=int, default=None,
                        help='Chunk index to process (for parallel processing)')
    parser.add_argument('--chunk_num', type=int, default=None,
                        help='Total number of chunks (for parallel processing)')
    args = parser.parse_args()
    
    
    # 100doh dir
    decode_dir   = os.path.join(args.clip_dir, '_decode')
    
    # Get all video files
    video_paths = glob.glob(f'{args.clip_dir}/*.mp4')
    video_paths.sort()  # Sort for consistent chunking
    
    # Handle chunking
    if args.chunk_idx is not None and args.chunk_num is not None:
        video_chunks = chunk_into_n(video_paths, args.chunk_num)
        if args.chunk_idx >= len(video_chunks):
            print(f"Chunk {args.chunk_idx} is out of range (total chunks: {len(video_chunks)})")
            exit(0)
        video_paths = video_chunks[args.chunk_idx]
        print(f"Processing chunk {args.chunk_idx}/{args.chunk_num-1} with {len(video_paths)} videos")
    else:
        print(f"Processing all {len(video_paths)} videos")
    
    for video_path in tqdm(video_paths):
        video_name = video_path.split('/')[-1][:-4]
        save_dir = os.path.join(decode_dir, video_name)
        decode_video(video_path, save_dir)