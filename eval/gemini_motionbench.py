import os
import json
import time
import concurrent.futures
from tqdm import tqdm
from google import genai
from google.genai import types
import fire

def load_video_info_from_jsonl(jsonl_path):
    """
    Load video and QA information from a JSONL file.
    
    Args:
        jsonl_path (str): Path to the JSONL file containing video info
        
    Returns:
        list: A list of dictionaries containing video and QA information
    """
    video_info_list = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                video_info = json.loads(line.strip())
                video_info_list.append(video_info)
    
    return video_info_list

def checkanswer(output, answer):
    a = answer.lower()
    o = output.lower().split(":")[0].split(",")[0].split(".")[0].strip()

    if a == "na":
        return 1
    
    if a in o:
        return 1
    else:
        return 0

def process_video_question(video_info, question, client, video_base_path, model_name, fps=4):
    """
    Process a single video question in parallel.
    
    Args:
        video_info (dict): Video information dictionary
        question (dict): Question dictionary
        client: Gemini client
        video_base_path (str): Base path for videos
        model_name (str): Name of the Gemini model to use
        fps (int): Frames per second for video metadata (default: 4)
        
    Returns:
        tuple: (video_path, question_info_dict)
    """
    video_path = os.path.join(video_base_path, video_info.get("video_path", ""))
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return video_path, None
    
    prompt = question["question"] + "\n" + "Please directly output the choice (A, B, C, D). No other text."
    
    try:
        video_bytes = open(video_path, 'rb').read()
        output = client.models.generate_content(
            model=model_name, 
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                        video_metadata=types.VideoMetadata(fps=fps)
                    ),
                    types.Part(text=prompt)
                ]
            ),
            config=types.GenerateContentConfig(
                # thinking_config=types.ThinkingConfig(thinking_budget=1024)
                # Turn off thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=0)
                # Turn on dynamic thinking:
                thinking_config=types.ThinkingConfig(thinking_budget=-1)
            ),

        ).text
        
        answer = question["answer"]
        score = checkanswer(output, answer)
        
        question_info = {
            "question": question["question"], 
            "prompt": prompt, 
            "answer": answer, 
            "output": output, 
            "score": score
        }
        
        print(f"Processed: {video_path} - Q: {question['question'][:50]}... - Score: {score}")
        return video_path, question_info
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return video_path, None

def save_results(output_json, output_file="gemini_answer_motionbench.json"):
    """Save results to JSON file with atomic write."""
    output_file_temp = output_file + ".temp"
    with open(output_file_temp, "w") as f:
        json.dump(output_json, f, indent=2)
    os.rename(output_file_temp, output_file)
    print(f"Saved results to {output_file}")

def main(max_workers=8, model="gemini-2.5-flash", motionbench_base_path=None, fps=4):
    if motionbench_base_path is None:
        # Download using Hugging Face if path not specified
        from pathlib import Path
        from huggingface_hub import snapshot_download

        print("motionbench_base_path not provided. Downloading MotionBench dataset from HuggingFace Hub ...")
        # Download the dataset snapshot to local_dir if not present
        local_dir =snapshot_download(
            repo_id="zai-org/MotionBench", 
            repo_type="dataset",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        motionbench_base_path = os.path.abspath(local_dir)
        print(f"MotionBench data downloaded to: {motionbench_base_path}")
    else:
        motionbench_base_path = os.path.abspath(motionbench_base_path)
        print(f"Using provided motionbench_base_path: {motionbench_base_path}")
        
    video_base_path = os.path.join(motionbench_base_path, "public-dataset")
        
    # Load video info from JSONL file
    video_info_path = os.path.join(motionbench_base_path, "MotionBench", "video_info.meta.jsonl") 
    video_info_list = load_video_info_from_jsonl(video_info_path)
    
    # Initialize Gemini client
    client = genai.Client()
    
    print(f"Found {len(video_info_list)} video entries")
    print(f"Using {max_workers} parallel workers")
    print(f"Using model: {model}")
    print(f"Video base path: {video_base_path}")
    print(f"Using fps: {fps}")
    
    # Generate output filename with model name
    model_suffix = model.replace("/", "_").replace("-", "_")
    output_file = f"gemini_answer_motionbench_{model_suffix}.json"
    
    # Prepare tasks for parallel processing
    tasks = []
    for video_info in video_info_list:
        video_path = os.path.join(video_base_path, video_info.get("video_path", ""))
        questions = video_info.get("qa", [])
        
        if not os.path.exists(video_path):
            print(f"Skipping entry with missing video: {video_path}")
            continue
            
        for question in questions:
            tasks.append((video_info, question, client, video_base_path, model, fps))
    
    print(f"Total tasks to process: {len(tasks)}")
    
    # Process results
    output_json = {}
    correct = 0
    total = 0
    
    # Process in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_video_question, *task): task 
            for task in tasks
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(tasks), desc="Processing videos") as pbar:
            for future in concurrent.futures.as_completed(future_to_task):
                video_path, question_info = future.result()
                
                if question_info is not None:
                    if video_path not in output_json:
                        output_json[video_path] = []
                    output_json[video_path].append(question_info)
                    
                    total += 1
                    correct += question_info["score"]
                    
                    # Update progress bar with current accuracy
                    pbar.set_postfix({
                        'accuracy': f"{correct/total:.3f}",
                        'correct': f"{correct}/{total}"
                    })
                
                pbar.update(1)
                
                # Save intermediate results every 10 completed tasks
                if total % 10 == 0 and total > 0:
                    save_results(output_json, output_file)
    
    # Add final statistics
    output_json["total_acc"] = {
        "accuracy": correct/total if total > 0 else 0, 
        "correct": correct, 
        "total": total
    }
    
    # Save final results
    save_results(output_json, output_file)
    
    print(f"\nFinal Results:")
    print(f"Total processed: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {correct/total:.3f}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
