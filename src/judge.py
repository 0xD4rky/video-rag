import torch
import cv2
import os
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from tqdm import tqdm
import re

class VideoJudge:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Initialize the Video Judge using BLIP model (much more memory efficient)
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Loading video understanding model on {self.device}...")
        
        # Load BLIP model and processor (much lighter than Qwen2-VL)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
        ).to(self.device)
        print("Video understanding model loaded successfully!")

    def extract_key_frames(self, video_path, start_time, end_time, num_frames=3):
        """
        Extract key frames from a video segment (reduced to 3 frames to save memory)
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
                # Resize to reduce memory usage
                frame_pil = Image.fromarray(frame).resize((224, 224))
                frames.append(frame_pil)
        
        cap.release()
        return frames

    def analyze_single_frame(self, frame, query):
        """
        Analyze a single frame and return description
        """
        try:
            # Generate caption for the frame
            inputs = self.processor(frame, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=100, num_beams=5)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return "Could not analyze frame"

    def calculate_relevance_score(self, descriptions, query):
        """
        Calculate relevance score based on text matching
        """
        query_words = set(query.lower().split())
        total_score = 0
        
        for desc in descriptions:
            desc_words = set(desc.lower().split())
            # Simple word overlap scoring
            common_words = query_words.intersection(desc_words)
            score = len(common_words) / max(len(query_words), 1)
            total_score += score
        
        # Average score and scale to 0-10
        avg_score = (total_score / max(len(descriptions), 1)) * 10
        return min(10.0, avg_score)

    def analyze_scene(self, video_path, start_time, end_time, query):
        """
        Analyze a single scene and return relevance score and explanation
        """
        # Extract key frames from the scene
        frames = self.extract_key_frames(video_path, start_time, end_time)
        
        if not frames:
            return 0.0, "No frames could be extracted from this scene."
        
        # Analyze each frame
        descriptions = []
        for i, frame in enumerate(frames):
            desc = self.analyze_single_frame(frame, query)
            descriptions.append(desc)
            print(f"    Frame {i+1}: {desc}")
        
        # Calculate relevance score
        relevance_score = self.calculate_relevance_score(descriptions, query)
        
        # Create explanation
        combined_description = " | ".join(descriptions)
        explanation = f"Scene contains: {combined_description}"
        
        # Simple keyword matching for additional scoring
        query_lower = query.lower()
        desc_lower = combined_description.lower()
        
        # Boost score if query keywords appear in descriptions
        query_keywords = query_lower.split()
        keyword_matches = sum(1 for keyword in query_keywords if keyword in desc_lower)
        keyword_boost = (keyword_matches / len(query_keywords)) * 3  # Up to 3 point boost
        
        final_score = min(10.0, relevance_score + keyword_boost)
        
        return final_score, explanation

    def judge_scenes(self, video_path, scenes_with_scores, query):
        """
        Judge multiple scenes and return the most relevant one
        """
        print(f"\nAnalyzing {len(scenes_with_scores)} scenes with video understanding model...")
        
        detailed_results = []
        
        for i, (start_time, end_time, clip_similarity) in enumerate(tqdm(scenes_with_scores, desc="Analyzing scenes")):
            print(f"\nAnalyzing Scene {i+1}: {start_time:.1f}s - {end_time:.1f}s")
            
            try:
                relevance_score, explanation = self.analyze_scene(video_path, start_time, end_time, query)
                
                detailed_results.append({
                    'scene_index': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'clip_similarity': clip_similarity,
                    'llm_relevance_score': relevance_score,
                    'explanation': explanation,
                    'combined_score': 0.6 * relevance_score + 0.4 * (clip_similarity * 10)
                })
                
                print(f"    LLM Relevance Score: {relevance_score:.2f}/10")
                print(f"    CLIP Similarity: {clip_similarity:.4f}")
                print(f"    Combined Score: {detailed_results[-1]['combined_score']:.2f}")
                
            except Exception as e:
                print(f"    Error analyzing scene: {e}")
                # Fallback to CLIP score only
                detailed_results.append({
                    'scene_index': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'clip_similarity': clip_similarity,
                    'llm_relevance_score': 5.0,  # Default score
                    'explanation': f"Error in analysis: {str(e)}",
                    'combined_score': clip_similarity * 10
                })

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
    try:
        judge = VideoJudge()
        
        # Analyze all scenes
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
        
    except Exception as e:
        print(f"Error in video judge: {e}")
        print("Falling back to CLIP scores only...")
        
        # Fallback: return the scene with highest CLIP score
        best_clip_scene = max(top_scenes, key=lambda x: x[2])
        return {
            'scene_index': 1,
            'start_time': best_clip_scene[0],
            'end_time': best_clip_scene[1],
            'clip_similarity': best_clip_scene[2],
            'llm_relevance_score': 5.0,
            'explanation': "Used CLIP similarity only due to model error",
            'combined_score': best_clip_scene[2] * 10
        }

# For standalone testing
if __name__ == "__main__":
    print("This module should be imported and used by embedding.py")
    print("Run embedding.py to use the complete video search pipeline with judge integration.")