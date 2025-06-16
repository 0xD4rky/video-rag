import os
from typing import List, Tuple
import numpy as np
from torch.nn.functional import cosine_similarity
import torch
import logging

logger = logging.getLogger(__name__)


def rank_scenes(scenes: List[str], query: str, scene_embeddings: np.ndarray = None, query_embedding: np.ndarray = None) -> List[Tuple[str, float]]:
    """
    ranking scenes using cosine similarity.
    Returns:
        List of (scene, relevance_score) tuples sorted by relevance
    """
    if scene_embeddings is not None and query_embedding is not None:
        return _rank_with_embeddings(scenes, scene_embeddings, query_embedding)
    else:
        return _fallback_ranking(scenes, query)


def _rank_with_embeddings(scenes: List[str], scene_embeddings: np.ndarray, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
    """ranking scenes using actual embeddings and cosine similarity."""
    ranked_scenes = []
    
    query_tensor = torch.tensor(query_embedding).unsqueeze(0)
    scene_tensors = torch.tensor(scene_embeddings)
    
    for i, scene_id in enumerate(scenes):
        scene_idx = int(scene_id)
        if scene_idx < len(scene_tensors):
            scene_tensor = scene_tensors[scene_idx].unsqueeze(0)
            similarity = cosine_similarity(query_tensor, scene_tensor).item()
            ranked_scenes.append((scene_id, similarity))
    
    ranked_scenes.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Ranked {len(ranked_scenes)} scenes using cosine similarity")
    
    return ranked_scenes


def _fallback_ranking(scenes: List[str], query: str) -> List[Tuple[str, float]]:
    """Fallback ranking using simple scoring with diversity penalty."""
    ranked_scenes = []
    
    for i, scene in enumerate(scenes):
        base_score = 1.0 / (i + 1)
        
        diversity_penalty = 0.0
        for j, other_scene in enumerate(scenes[:i]):
            if _scenes_similar(scene, other_scene):
                diversity_penalty += 0.1
        
        final_score = max(0.0, base_score - diversity_penalty)
        ranked_scenes.append((scene, final_score))
    
    ranked_scenes.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Ranked {len(ranked_scenes)} scenes using fallback method")
    
    return ranked_scenes


def _scenes_similar(scene1: str, scene2: str) -> bool:
    """Simple similarity check between scenes."""
    return scene1 == scene2