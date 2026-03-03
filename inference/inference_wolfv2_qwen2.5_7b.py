#!/usr/bin/env python3

import argparse
import logging
import time
import warnings
from typing import List
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:
    def process_vision_info(messages):
        image_inputs = []
        video_inputs = []
        for message in messages:
            for part in message.get("content", []):
                if part.get("type") == "image":
                    image_inputs.append(part.get("image") or part.get("url"))
                elif part.get("type") == "video":
                    video_inputs.append(part.get("video"))
        return image_inputs, video_inputs

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)


def elapsed_seconds(start_time: float) -> float:
    return time.perf_counter() - start_time


def _extract_with_cv2(video_path: str, max_frames: int) -> List[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames: List[Image.Image] = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Unable to read video: {video_path}")

    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frames


def _extract_with_imageio(video_path: str, max_frames: int) -> List[Image.Image]:
    import imageio.v3 as iio

    frames_np = iio.imread(video_path, index=None)
    if frames_np is None or len(frames_np) == 0:
        raise ValueError(f"Unable to read video: {video_path}")

    total_frames = len(frames_np)
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    frames: List[Image.Image] = []
    for frame_idx in frame_indices:
        frame = frames_np[frame_idx]
        if frame is None:
            continue
        frames.append(Image.fromarray(frame))

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frames


def _extract_with_torchvision(video_path: str, max_frames: int) -> List[Image.Image]:
    from torchvision.io import read_video

    video, _, _ = read_video(video_path, pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"Unable to read video: {video_path}")

    total_frames = video.shape[0]
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    frames: List[Image.Image] = []
    for frame_idx in frame_indices:
        frame = video[frame_idx].cpu().numpy()
        if frame is None:
            continue
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frames


def _get_video_duration_seconds(video_path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())

    ffmpeg_path = _find_ffmpeg_path()
    result = subprocess.run(
        [ffmpeg_path, "-i", video_path, "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s+(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
    if not match:
        raise ValueError("Unable to parse duration from ffmpeg output")
    hours, minutes, seconds, frac = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + float(f"0.{frac}")
    )


def _extract_with_ffmpeg(video_path: str, max_frames: int) -> List[Image.Image]:
    ffmpeg_path = _find_ffmpeg_path()
    try:
        duration = _get_video_duration_seconds(video_path)
    except Exception:
        duration = None

    frames: List[Image.Image] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        if duration and duration > 0:
            timestamps = np.linspace(0, max(duration - 1e-3, 0.0), min(max_frames, 64))
            for i, ts in enumerate(timestamps):
                out_path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
                subprocess.run(
                    [
                        ffmpeg_path,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{ts:.3f}",
                        "-i",
                        video_path,
                        "-frames:v",
                        "1",
                        out_path,
                    ],
                    check=True,
                )
        else:
            out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
            subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    video_path,
                    "-vf",
                    "fps=1",
                    "-frames:v",
                    str(max_frames),
                    out_pattern,
                ],
                check=True,
            )

        for name in sorted(os.listdir(tmpdir)):
            if name.lower().endswith(".jpg"):
                frames.append(Image.open(os.path.join(tmpdir, name)).convert("RGB"))

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frames


def extract_video_frames(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    try:
        return _extract_with_cv2(video_path, max_frames)
    except Exception as exc:
        exc_text = str(exc)
        if "libGL.so.1" in exc_text:
            logger.debug("cv2 unavailable (libGL missing). Falling back to imageio.")
        else:
            logger.warning("cv2 video decode failed (%s). Falling back to imageio.", exc)
    try:
        return _extract_with_torchvision(video_path, max_frames)
    except Exception:
        pass
    try:
        return _extract_with_imageio(video_path, max_frames)
    except Exception as exc:
        last_exc = exc
    try:
        return _extract_with_ffmpeg(video_path, max_frames)
    except Exception as exc:
        raise RuntimeError(
            "Failed to decode video. Install a backend like opencv-python, "
            "imageio[ffmpeg], torchvision with ffmpeg support, or ffmpeg/ffprobe."
        ) from exc


def _find_ffmpeg_path() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found in PATH and imageio-ffmpeg unavailable") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-to-text inference for Qwen2.5-VL models")
    parser.add_argument("--video", "-v", required=True, help="Path to input video")
    parser.add_argument(
        "--model",
        "-m",
        default="WoWolf/Qwen2_5vl-7b-fm-tuned",
        help="HF model name or local path",
    )
    parser.add_argument("--max-frames", type=int, default=8, help="Max frames sampled from video")
    parser.add_argument("--prompt", type=str, default="Describe the video.", help="Text prompt")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS metadata for video input")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    load_elapsed = elapsed_seconds(load_start)

    frames = extract_video_frames(args.video, max_frames=args.max_frames)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": args.fps},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    if not image_inputs:
        image_inputs = None
    if not video_inputs:
        video_inputs = [frames]

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    run_start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    gen_elapsed = elapsed_seconds(run_start)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    total_elapsed = elapsed_seconds(total_start)
    print(output_text.strip())
    print(
        f"********** load: {load_elapsed:.2f}s | run: {gen_elapsed:.2f}s | total: {total_elapsed:.2f}s **********"
    )


if __name__ == "__main__":
    main()
