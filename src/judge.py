import os
import torch
import cv2
import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
from tqdm import tqdm

class VideoJudge:

    def __init__(self, model_name = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initialize the Video Judge using Qwen2-VL model
        """

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"loading video understanding model on {self.device}")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Video understanding model loaded successfully!")

    def extract_key_frames(self, video_path, start_time, end_time, num_frames=5):
        """
        Extract key frames from a video segment
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame
        
        # Calculate frame indices to extract
        if total_frames <= num_frames:
            frame_indices = list(range(start_frame, end_frame))
        else:
            step = total_frames // num_frames
            frame_indices = [start_frame + i * step for i in range(num_frames)]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames
    
    def analyze_scene(self, video_path, start_time, end_time, query):
        """
        Analyze a single scene and return relevance score and explanation
        """

        frames = self.extract_key_frames(video_path, start_time, end_time)
        if not frames:
            return 0.0, "No frames could be extracted from this scene."
        

        prompt = f"""
        Analyze these video frames from a scene (timespan: {start_time:.1f}s - {end_time:.1f}s) and determine how relevant they are to the query: "{query}"

        Please provide:
        1. A relevance score from 0-10 (where 10 is perfectly relevant)
        2. A brief explanation of what you see in the frames
        3. How well the scene matches the query

        Respond in JSON format:
        {{
            "relevance_score": <score_0_to_10>,
            "description": "<what_you_see_in_frames>",
            "explanation": "<why_this_scene_matches_or_doesnt_match_the_query>"
        }}
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": prompt
                    }
                ] + [{"type": "image", "image": frame} for frame in frames[:3]]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad(): 
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.1)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        try:

            response = json.loads(output_text.strip())
            relevance_score = float(response.get("relevance_score", 0))
            description = response.get("description", "No description provided")
            explanation = response.get("explanation", "No explanation provided")

            return relevance_score, f"Description: {description}\nExplanation: {explanation}"

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing model response: {e}")
            print(f"Raw response: {output_text}")
            # Fallback: try to extract score from text
            lines = output_text.lower().split('\n')
            score = 5.0  # Default score
            for line in lines:
                if 'score' in line and any(char.isdigit() for char in line):
                    numbers = [float(s) for s in line.split() if s.replace('.', '').isdigit()]
                    if numbers:
                        score = min(10.0, max(0.0, numbers[0]))
                        break
            
            return score, output_text
        
    def judge_scenes(self, video_path, scenes_with_scores, query):

        print(f"\nAnalyzing {len(scenes_with_scores)} scenes using a video understanding llm")

        detailed_results = []
        
        for i, (start_time, end_time, clip_similarity) in enumerate(tqdm(scenes_with_scores, desc="Analyzing scenes")):
            print(f"\nAnalyzing Scene {i+1}: {start_time:.1f}s - {end_time:.1f}s")
            
            relevance_score, explanation = self.analyze_scene(video_path, start_time, end_time, query)
            
            detailed_results.append({
                'scene_index': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'clip_similarity': clip_similarity,
                'llm_relevance_score': relevance_score,
                'explanation': explanation,
                'combined_score': 0.6 * relevance_score + 0.4 * (clip_similarity * 10)  # Combine LLM score with CLIP score
            })
            
            print(f"LLM Relevance Score: {relevance_score:.2f}/10")
            print(f"CLIP Similarity: {clip_similarity:.4f}")
            print(f"Combined Score: {detailed_results[-1]['combined_score']:.2f}")
            print(f"Analysis: {explanation}")

        # Sort by combined score
        detailed_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return detailed_results

def select_best_scene(video_path, top_scenes, query):
    """
    Main function to select the best scene using video understanding
    
    Args:
        video_path: Path to the original video
        top_scenes: List of tuples (start_time, end_time, similarity_score) from CLIP
        query: User's search query
    
    Returns:
        Dictionary with details of the best scene
    """
    judge = VideoJudge()
    
    results = judge.judge_scenes(video_path, top_scenes, query)
    
    if not results:
        print("No scenes could be analyzed!")
        return None
    
    best_scene = results[0]
    
    print(f"\n{'='*60}")
    print("FINAL JUDGMENT - MOST RELEVANT SCENE:")
    print(f"{'='*60}")
    print(f"Scene {best_scene['scene_index']}: {best_scene['start_time']:.1f}s - {best_scene['end_time']:.1f}s")
    print(f"LLM Relevance Score: {best_scene['llm_relevance_score']:.2f}/10")
    print(f"CLIP Similarity Score: {best_scene['clip_similarity']:.4f}")
    print(f"Combined Score: {best_scene['combined_score']:.2f}")
    print(f"\nAnalysis:")
    print(best_scene['explanation'])
    print(f"{'='*60}")
    
    # Also show rankings of all scenes
    print(f"\nAll Scenes Ranked:")
    for i, scene in enumerate(results):
        print(f"{i+1}. Scene {scene['scene_index']}: {scene['start_time']:.1f}s-{scene['end_time']:.1f}s "
              f"(Combined: {scene['combined_score']:.2f}, LLM: {scene['llm_relevance_score']:.2f}, "
              f"CLIP: {scene['clip_similarity']:.4f})")
    
    return best_scene

if __name__ == "__main__":
    # Test data - replace with actual scene data
    video_path = "test_video.mp4"
    query = "person walking with a dog"
    top_scenes = [
        (10.0, 15.0, 0.85),  # (start_time, end_time, clip_similarity)
        (25.0, 30.0, 0.82),
        (45.0, 50.0, 0.78)
    ]
    
    best_scene = select_best_scene(video_path, top_scenes, query)
    if best_scene:
        print(f"Best scene selected: {best_scene['start_time']:.1f}s - {best_scene['end_time']:.1f}s")


        


