import os
from typing import List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logging.basicConfig(
    filename="/Users/darky/Documents/video-rag/data/logs/judge.log",
    format='%(asctime)s %(message)s',
    filemode='w'
)

logger = logging.getLogger()

class Judge():

    def __init__(self, model_name: str):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )   
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def rank_scenes(self, scenes: List[str], query: str) -> List[Tuple[str, float]]:
        """
        Rank scenes using video LLM.
        
        Args:
            scenes: List of scene descriptions
            query: Search query
            
        Returns:
            List of (scene_description, relevance_score) tuples
        """
        ranked_scenes = []
        
        for scene_desc in scenes:
            prompt = f"""
            You are a very intelligent judge.
            Analyze the scene and the query in depth, and then carefully rate how well this video scene matches the search query on a scale of 0.0 to 1.0 (include decimal based rating).
            Only respond with a number.
            
            Query: {query}
            Scene: {scene_desc}
            
            Relevance score:
            """
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            try:
                score = float(response.split()[0])
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except (ValueError, IndexError):
                score = 0.0
            
            ranked_scenes.append((scene_desc, score))

        ranked_scenes.sort(key=lambda x: x[1], reverse=True)
        return ranked_scenes
    

def rank_scenes(scenes: List[str], query: str) -> List[Tuple[str, float]]:
    """
    Main scene ranking function with fallback logic.
    
    Args:
        scenes: List of scene descriptions or identifiers
        query: Search query
        
    Returns:
        List of (scene, relevance_score) tuples sorted by relevance
    """
    video_llm_model = os.getenv("VIDEO_LLM_MODEL")
    
    if video_llm_model:
        try:
            ranker = Judge(video_llm_model)
            return ranker.rank_scenes(scenes, query)
        except Exception as e:
            logger.warning(f"Video LLM ranking failed: {e}. Falling back to CLIP scoring.")
    
    return _fallback_clip_ranking(scenes, query)

def _fallback_clip_ranking(scenes: List[str], query: str) -> List[Tuple[str, float]]:
    """
    Fallback ranking using CLIP similarity and diversity penalty.
    
    Args:
        scenes: List of scene identifiers
        query: Search query
        
    Returns:
        List of (scene, relevance_score) tuples
    """
    # incase of fallback, we'll use scene indices as scores
    # this would typically involve computing CLIP similarities
    # but for simplicity, we'll use a placeholder scoring mechanism
    
    ranked_scenes = []
    for i, scene in enumerate(scenes):
        # Placeholder scoring: inverse of index with some randomness
        base_score = 1.0 / (i + 1)
        
        # Add diversity penalty for similar scenes (simplified)
        diversity_penalty = 0.0
        for j, other_scene in enumerate(scenes[:i]):
            if _scenes_similar(scene, other_scene):
                diversity_penalty += 0.1
        
        final_score = max(0.0, base_score - diversity_penalty)
        ranked_scenes.append((scene, final_score))
    
    # Sort by score descending
    ranked_scenes.sort(key=lambda x: x[1], reverse=True)
    return ranked_scenes


def _scenes_similar(scene1: str, scene2: str) -> bool:
    """
    Simple similarity check between scenes.
    
    Args:
        scene1: First scene identifier
        scene2: Second scene identifier
        
    Returns:
        True if scenes are considered similar
    """
    return scene1 == scene2