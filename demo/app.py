import os
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_ID = "WoWolf/Qwen2_5vl-7b-fm-tuned"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

def extract_video_frames(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Extract key frames from video using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return frames
    
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

SYSTEM_PROMPT = (
    "You are a helpful assistant that watches a user-provided video and answers "
    "questions about it concisely and accurately."
)

def build_messages(frames: List[Image.Image], question: str, fps: float = 1.0):
    """Build messages in Qwen-VL format"""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,  # Directly pass the frame list
                    "fps": fps,
                },
                {"type": "text", "text": question},
            ],
        },
    ]
    return messages


@torch.inference_mode()
def answer(video, question, max_frames=8, max_new_tokens=256, temperature=0.0):
    if video is None:
        return "Please upload a video first."
    if not question or question.strip() == "":
        question = "Describe this video in detail."

    # Extract frames from video
    frames = extract_video_frames(video, max_frames=int(max_frames))
    if not frames:
        return "Error: Unable to extract frames from video."

    # Build messages
    messages = build_messages(frames, question, fps=1.0)

    # Apply chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Process vision info (key step from qwen_vl_utils)
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generation settings
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=(float(temperature) > 0.0),
        temperature=float(temperature) if float(temperature) > 0 else None,
        pad_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )

    generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


with gr.Blocks(title="Video Q&A with Qwen2.5-VL-3B") as demo:
    gr.Markdown(
        """
        # FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos
        Upload a video, ask a question, and get an answer!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Upload Video (mp4, mov, webm)")
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is happening in this video?",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")
        
        with gr.Column(scale=1):
            output = gr.Textbox(label="Answer", lines=12, show_copy_button=True)

    with gr.Accordion("⚙️ Advanced Settings", open=False):
        with gr.Row():
            max_frames = gr.Slider(4, 16, value=8, step=1, label="Max Frames to Extract")
            max_new_tokens = gr.Slider(64, 512, value=256, step=32, label="Max New Tokens")
        with gr.Row():
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature (0=deterministic)")

    gr.Examples(
        examples=[
            ["What is happening in this video?"],
            ["Describe the main objects and actions in this video."],
            ["Summarize this video in a few sentences."],
        ],
        inputs=[question],
    )

    ask_btn.click(
        fn=answer,
        inputs=[video, question, max_frames, max_new_tokens, temperature],
        outputs=[output],
    )


if __name__ == "__main__":
    demo.launch()