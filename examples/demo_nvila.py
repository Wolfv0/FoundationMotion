#!/usr/bin/env python3
"""
Standalone script to evaluate NVILA model.
"""

import argparse
import os
import sys
import llava
from llava.media import Video

def main():
    parser = argparse.ArgumentParser(description="Run inference with NVILA-8B-Video")
    parser.add_argument("--model_path", type=str, default="", 
                        help="Path to the model")
    parser.add_argument("--video_path", type=str, required=True, 
                        help="Path to the input video file")
    parser.add_argument("--prompt", type=str, required=True, 
                        help="Text prompt/question for the model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        sys.exit(1)
    
    print(f"Loading model: {args.model_path}")
    model = llava.load(args.model_path)
    print("Model loaded successfully.")
    
    # Prepare video input
    print(f"Loading video: {args.video_path}")
    media = Video(args.video_path)
    
    # Generate response
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)
    print("Generating response...")
    
    output = model.generate_content([media, args.prompt])

    print("\nModel Response:")
    print(output)
    print("-" * 60)

if __name__ == "__main__":
    main()

