import os
import json
import fire

def enumerate_mp4_files(directory):
    """
    Enumerate all .mp4 files under the specified directory.
    
    Args:
        directory (str): The directory to search for .mp4 files
        
    Returns:
        list: A list of paths to .mp4 files
    """
    mp4_files = []
    print("Enumerating .mp4 files in directory:", directory)
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Filter for .mp4 files
        for file in files:
            if file.lower().endswith('.mp4'):
                # Get the full path
                full_path = os.path.join(root, file)
                mp4_files.append(full_path)
    
    return mp4_files

def main(task="av_hands_eval", model_path = "Efficient-Large-Model/NVILA-Lite-15B-Video"):
    """
    Main function to process video files and generate answers using NVILA model.
    
    Args:
        task (str): The task name used to construct directory paths. Default is "av_hands".
    """
    model_name = os.path.basename(model_path)
    output_dir = f"wolfv2eval/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"wolfv2eval/{model_name}/{task}.json"
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipping...")
        return

    # Directory to search
    video_dir = f"/home/ligengz/workspace/v2-dev/{task}/videos"
    qa_dir = f"/home/ligengz/workspace/v2-dev/{task}/qa_shuffled"

    # Get all mp4 files
    mp4_files = enumerate_mp4_files(video_dir)

    from llava.media import Image, Video
    import llava
    # from llava import conversation as clib
    # from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
    # model_path = "Efficient-Large-Model/NVILA-Lite-8B"
    # model_path = "Efficient-Large-Model/NVILA-8B-Video"
    # model_path = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/models/nvila-internal-33b-video-v1"
    
    
    model = None
    # Print the results
    print(f"Found {len(mp4_files)} .mp4 files:")

    def checkanswer(output, answer):
        a = answer.lower()
        o = output.lower().split(":")[0].split(",")[0].split(".")[0].strip()
        if a in o:
            return 1
        else:
            return 0

    '''
    {
      "gpt4o_mini_res": {
        "questions": [
          {
            "question": "What is the man doing with his hands?",
            "A": "Using one hand to gesture.",
            "B": "Both hands are not visible and not interacting with any objects.",
            "C": "Holding an object in his right hand.",
            "D": "Holding an object in his left hand.",
            "correct_answer": "B"
          },
          {
            "question": "Which hand is the woman using to hold another object in front of her?",
            "A": "Her left hand.",
            "B": "Neither hand.",
            "C": "Her right hand.",
            "D": "Both hands.",
            "correct_answer": "C"
          }
        ]
      }
    }
    '''
    output_json = {}
    correct = 0
    total = 0 
    for file_path in mp4_files:
        json_path = file_path.replace(video_dir, qa_dir).replace(".mp4", ".json")
        print("mp4 file:", file_path, "json file:", json_path)
        j = json.load(open(json_path))

        if "human" in j:
            questions = j["human"]["questions"]# [0]
        else:
            questions = j["gpt4o_mini_res"]["questions"]

        info = []
        for question in questions:
            prompt = question["question"] + "\n" + "\n".join([f"{k}: {question[k]}" for k in ["A", "B", "C", "D"]])
            print(prompt)
            media = Video(file_path)
            if model is None:
                print(f"loading {model_path}")
                model = llava.load(model_path)
                print("loading model done")
            output = model.generate_content([media, prompt])
            # TODO: this should be fixed in later revision
            if "correct_answer" in question:
                answer = question["correct_answer"]
            else:
                answer = question["answer"]
            score = checkanswer(output, answer)
            total += 1
            correct += score
            print(answer, output, "score:", score, f"accuracy: {correct/total:.2f}", f"total: {correct}/{total}")
            info.append({"question": question["question"], "answer": answer, "output": output, "score": score})
 
        # Only add the summary once, at the beginning
        output_json["__summary__"] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total
        }
        
        output_json[file_path] = info
        with open(output_file + ".tmp", "w") as f:
            json.dump(output_json, f, indent=2)

    os.rename(output_file + ".tmp", output_file)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)