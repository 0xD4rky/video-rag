import torch
import cv2
import os
import json
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import numpy as np
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from typing import List, Tuple, Dict
import gc
from dotenv import load_dotenv



class VideoJudge:
    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
        """
        Initialize the Video Judge using BLIP model (much more memory efficient)
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Loading video understanding model on {self.device}...")

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # trying llava instead of blip but using a fallback in case blip fails
        try:
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            
            # enabling memory efficient attention if available
            if hasattr(self.model, 'enable_memory_efficient_attention'):
                self.model.enable_memory_efficient_attention()
            
            print("LLaVA model loaded with optimizations")
            self.use_llava = True

        except Exception as e:
            print(f"Error loading LLaVA model: {e}")
            print("Falling back to BLIP model...")
            # Fallback to original blip model
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            ).to(self.device)
            self.use_llava = False
        print("Video understanding model loaded successfully!")

        print("Loading sentence transformer for semantic similarity...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformer loaded!")

        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            print("Gemini model configured for query expansion!")
        else:
            self.gemini_model = None
            print("Warning: No Gemini API key provided. Query expansion will be skipped.")


    def clear_memory(self):
        """
        Clear GPU memory to prevent OOM errors
        """
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def expand_query_with_gemini(self, query: str) -> str:
        if not self.gemini_model:
            return query
        
        try:

            expansion_prompt = f"""
            You are helping to expand a search query for video analysis. The query will be used to find relevant scenes in a video.
            
            Original query: "{query}"
            
            Please expand this query by:
            1. Adding synonyms and related terms
            2. Including visual descriptions that might appear in relevant scenes
            3. Adding context about what actions, objects, or situations might be present
            4. Keeping it concise but comprehensive
            
            Return only the expanded query without explanation.
            
            Example:
            Original: "person cooking"
            Expanded: "person cooking food, chef preparing meal, kitchen activities, cutting vegetables, stirring pot, using stove, culinary preparation, food preparation, cooking utensils, kitchen tools"
            
            Now expand: "{query}"
            """

            response = self.gemini_model.generate_content(expansion_prompt)
            expanded_query = response.text.strip()
            print(f"Original query: {query}")
            print(f"Expanded query: {expanded_query}")
            
            return expanded_query

        except Exception as e:
            print(f"Error expanding query with Gemini: {e}")
            return query

    def extract_key_frames(self, video_path, start_time, end_time, num_frames=4):
        """
        Extract key frames from a video segment with adaptive sampling
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame
        
        # Adaptive frame selection based on segment length
        if total_frames <= num_frames:
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Sample frames more intelligently - beginning, middle, end, and one random
            quarter = total_frames // 4
            frame_indices = [
                start_frame,  # Beginning
                start_frame + quarter,  # First quarter
                start_frame + 2 * quarter,  # Middle
                start_frame + 3 * quarter,  # Third quarter
            ]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to optimal size for LLaVA (336x336 is recommended)
                frame_pil = Image.fromarray(frame).resize((336, 336))
                frames.append(frame_pil)
        
        cap.release()
        return frames

    def analyze_single_frame(self, frame, query):
        """
        Analyze a single frame and return description
        """
        try:
            if self.use_llava:
                # Use LLaVA with detailed prompting
                prompt = f"""<image>USER: Look at this image and describe what you see in detail, focusing on elements related to: {expanded_query}

Pay attention to:
- Objects and people present
- Actions being performed
- Setting and environment
- Any relevant details that match the query

Provide a detailed description.
ASSISTANT:"""
                
                inputs = self.processor(prompt, frame, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs, 
                        max_length=100, 
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                response = self.processor.decode(output[0], skip_special_tokens=True)

                if "ASSISTANT:" in response:
                    description = response.split("ASSISTANT:")[-1].strip()
                else:
                    description = response.strip()
            
            else:
                inputs = self.processor(frame, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_length=100, num_beams=5)
                    description = self.processor.decode(out[0], skip_special_tokens=True)

            self.clear_memory() # clearing memory after each inference
            return description
            
         
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            self.clear_memory()
            return "Could not analyze frame"
        
    def calculate_semantic_similarity(self, descriptions: List[str], expanded_query: str) -> float:
        """
        Calculate semantic similarity using sentence transformers
        """
        try:
            # Encode descriptions and query
            description_embeddings = self.sentence_model.encode(descriptions)
            query_embedding = self.sentence_model.encode([expanded_query])
            
            # Calculate cosine similarities
            similarities = util.cos_sim(query_embedding, description_embeddings)[0]
            
            # Return average similarity scaled to 0-10
            avg_similarity = float(similarities.mean())
            return min(10.0, max(0.0, avg_similarity * 10))
            
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 5.0

    def calculate_relevance_score(self, descriptions: List[str], query: str, expanded_query: str) -> Tuple[float, Dict]:
        """
        Calculate enhanced relevance score combining multiple methods
        """
        # 1. Semantic similarity score (primary)
        semantic_score = self.calculate_semantic_similarity(descriptions, expanded_query)
        
        # 2. Keyword matching score (secondary)
        query_words = set(query.lower().split())
        expanded_words = set(expanded_query.lower().split())
        all_query_words = query_words.union(expanded_words)
        
        keyword_scores = []
        for desc in descriptions:
            desc_words = set(desc.lower().split())
            common_words = all_query_words.intersection(desc_words)
            score = len(common_words) / max(len(all_query_words), 1)
            keyword_scores.append(score)
        
        avg_keyword_score = sum(keyword_scores) / max(len(keyword_scores), 1) * 10
        
        # 3. Content density score (how detailed are the descriptions)
        avg_description_length = sum(len(desc.split()) for desc in descriptions) / max(len(descriptions), 1)
        density_score = min(10.0, avg_description_length / 10)  # Normalize to 0-10
        
        # Combine scores with weights
        final_score = (
            0.6 * semantic_score +      # Primary: semantic understanding
            0.3 * avg_keyword_score +   # Secondary: keyword matching
            0.1 * density_score         # Tertiary: content richness
        )
        
        score_breakdown = {
            'semantic_score': semantic_score,
            'keyword_score': avg_keyword_score,
            'density_score': density_score,
            'final_score': min(10.0, final_score)
        }
        
        return min(10.0, final_score), score_breakdown

    def analyze_scene(self, video_path, start_time, end_time, query, expanded_query):
        """
        Analyze a single scene with enhanced understanding
        """
        # Extract key frames from the scene
        frames = self.extract_key_frames(video_path, start_time, end_time)
        
        if not frames:
            return 0.0, "No frames could be extracted from this scene.", {}
        
        # Analyze each frame
        descriptions = []
        for i, frame in enumerate(frames):
            desc = self.analyze_single_frame(frame, expanded_query)
            descriptions.append(desc)
            print(f"    Frame {i+1}: {desc[:100]}...")  # Truncate for display
        
        # Calculate enhanced relevance score
        relevance_score, score_breakdown = self.calculate_enhanced_relevance_score(
            descriptions, query, expanded_query
        )
        
        combined_description = " | ".join(descriptions)
        explanation = f"Scene analysis: {combined_description}"
        
        return relevance_score, explanation, score_breakdown

    def judge_scenes(self, video_path, scenes_with_scores, query):
        """
        Judge multiple scenes with enhanced analysis
        """
        print(f"\nExpanding query with Gemini...")
        expanded_query = self.expand_query_with_gemini(query)
        
        print(f"\nAnalyzing {len(scenes_with_scores)} scenes with enhanced video understanding...")
        
        detailed_results = []
        
        for i, (start_time, end_time, clip_similarity) in enumerate(tqdm(scenes_with_scores, desc="Analyzing scenes")):
            print(f"\nAnalyzing Scene {i+1}: {start_time:.1f}s - {end_time:.1f}s")
            
            try:
                relevance_score, explanation, score_breakdown = self.analyze_scene(
                    video_path, start_time, end_time, query, expanded_query
                )
                
                # Enhanced combined score calculation
                combined_score = (
                    0.7 * relevance_score +           # Higher weight on LLM analysis
                    0.3 * (clip_similarity * 10)      # Lower weight on CLIP
                )
                
                detailed_results.append({
                    'scene_index': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'clip_similarity': clip_similarity,
                    'llm_relevance_score': relevance_score,
                    'score_breakdown': score_breakdown,
                    'explanation': explanation,
                    'combined_score': combined_score,
                    'expanded_query': expanded_query
                })
                
                print(f"    LLM Relevance Score: {relevance_score:.2f}/10")
                print(f"      - Semantic: {score_breakdown['semantic_score']:.2f}")
                print(f"      - Keyword: {score_breakdown['keyword_score']:.2f}")
                print(f"      - Density: {score_breakdown['density_score']:.2f}")
                print(f"    CLIP Similarity: {clip_similarity:.4f}")
                print(f"    Combined Score: {combined_score:.2f}")
                
            except Exception as e:
                print(f"    Error analyzing scene: {e}")
                self.clear_memory()
                # Fallback to CLIP score only
                detailed_results.append({
                    'scene_index': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'clip_similarity': clip_similarity,
                    'llm_relevance_score': 5.0,
                    'score_breakdown': {'semantic_score': 5.0, 'keyword_score': 5.0, 'density_score': 5.0, 'final_score': 5.0},
                    'explanation': f"Error in analysis: {str(e)}",
                    'combined_score': clip_similarity * 10,
                    'expanded_query': expanded_query
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