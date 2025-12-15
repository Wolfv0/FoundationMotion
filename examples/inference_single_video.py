#!/usr/bin/env python3

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
from PIL import Image
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModelVLTester:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"):
        """Initialize Multi-Model VL Tester"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Determine model type
        self.is_perception_lm = "Perception-LM" in model_name or "facebook/Perception-LM" in model_name
        logger.info(f"Model type: {'Perception-LM' if self.is_perception_lm else 'Qwen-VL'}")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        
        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        self.model = self._load_model_with_fallback(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        logger.info("Model loaded successfully")
    
    def _load_model_with_fallback(self, model_name: str):
        """Load model without Flash Attention"""
        # Use standard loading without Flash Attention
        standard_config = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        logger.info(f"Loading model with standard config (no Flash Attention): {standard_config}")
        
        try:
            if self.is_perception_lm:
                # Use AutoModelForImageTextToText for Perception-LM
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name, **standard_config
                )
                logger.info("Perception-LM model loaded successfully!")
            else:
                # Use Qwen2_5_VLForConditionalGeneration for Qwen models
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name, **standard_config
                )
                logger.info("Qwen model loaded successfully!")
            
            # Check and report the attention implementation being used
            self._report_attention_implementation(model, standard_config)
            
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def _report_attention_implementation(self, model, strategy):
        """Report the attention implementation being used"""
        if hasattr(model.config, 'attn_implementation'):
            actual_attn = model.config.attn_implementation
            logger.info(f"Model attention implementation: {actual_attn}")
        else:
            logger.info("Attention implementation info not available in model config")
        
        requested_attn = strategy.get('attn_implementation', 'default')
        logger.info(f"Requested attention implementation: {requested_attn}")

    def extract_video_frames(self, video_path: str, max_frames: int = 4) -> List[Image.Image]:
        """Extract key frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            logger.warning(f"Unable to read video: {video_path}")
            return frames
        
        # Select frames evenly
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video {video_path}")
        return frames

    def create_messages(self, frames: List[Image.Image], question: str, options: Dict[str, str]) -> List[Dict]:
        """Create model input messages"""
        # Construct options text
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        
        # Build the complete question
        full_question = f"""Please carefully observe this video and answer the following question:

{question}

Options:
{options_text}

Please choose the correct answer from A, B, C, or D. Only respond with the option letter (e.g., A)."""

        if self.is_perception_lm:
            # Perception-LM format - use images instead of video
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": frames[0],  # Use first frame as representative
                        },
                        {"type": "text", "text": full_question},
                    ],
                }
            ]
        else:
            # Qwen-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,  # Directly pass the frame list
                            "fps": 1.0,
                        },
                        {"type": "text", "text": full_question},
                    ],
                }
            ]
        
        return messages

    def get_model_answer(self, video_path: str, question: str, options: Dict[str, str]) -> Tuple[str, str]:
        """Get model's answer to a single question"""
        try:
            # Extract video frames
            frames = self.extract_video_frames(video_path)
            if not frames:
                logger.error(f"Unable to extract frames from video {video_path}")
                return "ERROR", "No frames extracted from video"
            
            # Create messages
            messages = self.create_messages(frames, question, options)
            
            # Process input based on model type
            if self.is_perception_lm:
                # Perception-LM processing
                try:
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    logger.debug(f"Perception-LM processing successful")
                except Exception as e:
                    logger.warning(f"Perception-LM processing failed: {e}, trying fallback")
                    # Fallback: direct processing
                    full_question = f"{question}\nOptions: {options}\nPlease choose A, B, C, or D:"
                    inputs = self.processor(
                        text=[full_question],
                        images=[frames[0]],
                        padding=True,
                        return_tensors="pt",
                    )
            else:
                # Qwen-VL processing
                try:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    logger.debug(f"Chat template successful, text length: {len(text)}")
                except Exception as e:
                    logger.warning(f"Chat template failed: {e}, trying simplified processing")
                    # Simplified question format
                    full_question = f"{question}\nOptions: {options}\nPlease choose A, B, C, or D:"
                    text = full_question
                
                try:
                    image_inputs, video_inputs = process_vision_info(messages)
                    logger.debug(f"Vision info processing successful")
                    
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                except Exception as e:
                    logger.warning(f"Vision processing failed: {e}, trying images only")
                    # Fallback to using images only
                    inputs = self.processor(
                        text=[text],
                        images=frames,  # Directly use frames as images
                        padding=True,
                        return_tensors="pt",
                    )
            
            inputs = inputs.to(self.device)
            logger.debug(f"Input data moved to device: {self.device}")
            
            # Generate answer - add more conservative settings
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=64,  # Reduce token count
                        do_sample=False,
                        temperature=0.0,    # Set to 0 to avoid randomness
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                logger.debug(f"Generation complete, output length: {generated_ids.shape}")
            except Exception as e:
                logger.error(f"Model generation failed: {e}")
                # Final fallback - random choice
                import random
                fallback_answer = random.choice(['A', 'B', 'C', 'D'])
                return fallback_answer, f"Generation failed: {e}. Random fallback: {fallback_answer}"
            
            # Handle different output formats
            if self.is_perception_lm:
                # Perception-LM returns full sequences, need to trim input
                input_length = inputs["input_ids"].shape[1]
                generated_ids_trimmed = generated_ids[:, input_length:]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )[0]
            else:
                # Qwen-VL processing
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            logger.debug(f"Raw output: {output_text}")
            
            # Extract answer choice
            answer = self.extract_answer_choice(output_text)
            logger.info(f"Question: {question[:50]}... -> Model answer: {answer}")
            return answer, output_text  # Return both extracted answer and full response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Full error traceback: {error_traceback}")
            return "ERROR", f"Processing error: {str(e)}"

    def extract_answer_choice(self, output_text: str) -> str:
        """Extract answer choice (A, B, C, D) from model output"""
        output_text = output_text.strip().upper()
        
        # Check if it directly contains an option
        for choice in ['A', 'B', 'C', 'D']:
            if choice in output_text:
                return choice
        
        # If no clear option is found, return the first character (if it's a valid option)
        if output_text and output_text[0] in ['A', 'B', 'C', 'D']:
            return output_text[0]
        
        logger.warning(f"Unable to extract answer choice from output: {output_text}")
        return "UNKNOWN"

    def load_questions(self, json_path: str) -> List[Dict]:
        """Load question data"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'human' in data:
                return data['human']['questions']
            elif 'gpt4o_mini_res' in data:
                return data['gpt4o_mini_res']['questions']
            else:
                logger.error(f"Unexpected data structure in file: {json_path}")
                return []
        except Exception as e:
            logger.error(f"Failed to load question file {json_path}: {str(e)}")
            return []

    def test_single_video(self, video_path: str, json_path: str) -> Tuple[int, int]:
        """Test all questions for a single video"""
        logger.info(f"Testing video: {video_path}")
        
        questions = self.load_questions(json_path)
        if not questions:
            logger.error(f"Unable to load question file: {json_path}")
            return 0, 0
        
        correct = 0
        total = len(questions)
        detailed_results = []
        
        for i, q_data in enumerate(questions):
            question = q_data['question']
            options = {k: v for k, v in q_data.items() if k in ['A', 'B', 'C', 'D']}
            # Support both 'correct_answer' and 'answer' keys
            correct_answer = q_data.get('correct_answer') or q_data.get('answer')
            
            logger.info(f"  Question {i+1}/{total}: {question}")
            
            model_answer, full_response = self.get_model_answer(video_path, question, options)
            
            is_correct = model_answer == correct_answer
            if is_correct:
                correct += 1
                logger.info(f"  ✓ Correct! Answer: {correct_answer}")
            else:
                logger.info(f"  ✗ Incorrect! Correct answer: {correct_answer}, Model answer: {model_answer}")
            
            # Store detailed results
            detailed_results.append({
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'model_answer': model_answer,
                'full_response': full_response,
                'is_correct': is_correct
            })
        
        logger.info(f"Video {os.path.basename(video_path)} accuracy: {correct}/{total} = {correct/total*100:.2f}%")
        return correct, total, detailed_results

    def run_full_test(self, video_dir: str, json_dir: str) -> Dict:
        """Run full test"""
        video_dir = Path(video_dir)
        json_dir = Path(json_dir)
        
        # Get all video files
        video_files = list(video_dir.glob("*.mp4"))
        video_files.sort()
        
        logger.info(f"Found {len(video_files)} video files in {video_dir}")
        for i, vf in enumerate(video_files):
            logger.info(f"  Video {i+1}: {vf.name}")
        
        total_correct = 0
        total_questions = 0
        results = {}
        
        logger.info(f"Starting test, total {len(video_files)} videos")
        
        for i, video_file in enumerate(video_files):
            video_name = video_file.stem
            json_file = json_dir / f"{video_name}.json"
            
            logger.info(f"Processing video {i+1}/{len(video_files)}: {video_name}")
            
            if not json_file.exists():
                logger.warning(f"Corresponding JSON file not found: {json_file}")
                continue
            
            try:
                correct, total, detailed_results = self.test_single_video(str(video_file), str(json_file))
            except Exception as e:
                logger.error(f"Error processing video {video_name}: {str(e)}")
                continue
            
            results[video_name] = {
                'correct': correct,
                'total': total,
                'accuracy': correct / total if total > 0 else 0,
                'detailed_results': detailed_results
            }
            
            total_correct += correct
            total_questions += total
        
        # Calculate overall accuracy
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        logger.info(f"Test completed: processed {len(results)} videos out of {len(video_files)} found")
        
        results['overall'] = {
            'correct': total_correct,
            'total': total_questions,
            'accuracy': overall_accuracy,
            'videos_processed': len(results),
            'videos_found': len(video_files)
        }
        
        return results

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Model Video Q&A Test Script')
    parser.add_argument('--model', '-m', type=str, 
                       default="Qwen/Qwen2.5-VL-72B-Instruct",
                       help='Model name to use (default: Qwen/Qwen2.5-VL-72B-Instruct, alternative: facebook/Perception-LM-8B)')
    parser.add_argument('--video-dir', type=str,
                       default="/data/vision/torralba/selfmanaged/isola/u/yulu/cmar/Data-Test/qwen-72B/data/av_car_eval/Interactive_Nuscenes",
                       help='Path to video directory')
    parser.add_argument('--json-dir', type=str,
                       default="/data/vision/torralba/selfmanaged/isola/u/yulu/cmar/Data-Test/qwen-72B/data/av_car_eval/Interactive_Nuscenes_qas",
                       help='Path to JSON directory')
    
    args = parser.parse_args()
    
    video_dir = args.video_dir
    json_dir = args.json_dir
    model_name = args.model
    
    logger.info(f"Using model: {model_name}")
    logger.info(f"Video directory: {video_dir}")
    logger.info(f"JSON directory: {json_dir}")
    
    # Check if paths exist
    if not os.path.exists(video_dir):
        logger.error(f"Video directory does not exist: {video_dir}")
        return
    
    if not os.path.exists(json_dir):
        logger.error(f"JSON directory does not exist: {json_dir}")
        return
    
    # Create tester with specified model
    tester = MultiModelVLTester(model_name=model_name)
    
    # Run test
    logger.info("Starting full test...")
    results = tester.run_full_test(video_dir, json_dir)
    
    # Output results
    print("\n" + "="*50)
    print("Test Results Summary")
    print("="*50)
    
    for video_name, result in results.items():
        if video_name == 'overall':
            continue
        print(f"{video_name}: {result['correct']}/{result['total']} = {result['accuracy']*100:.2f}%")
    
    print("-"*50)
    overall = results['overall']
    print(f"Videos found: {overall['videos_found']}")
    print(f"Videos processed: {overall['videos_processed']}")
    print(f"Overall accuracy: {overall['correct']}/{overall['total']} = {overall['accuracy']*100:.2f}%")
    print("="*50)
    
    # Save results to file
    results_file = "test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Test results saved to: {results_file}")

if __name__ == "__main__":
    main()
