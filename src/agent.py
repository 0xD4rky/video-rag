"""Main video search agent orchestrator."""

import os
import sys
import asyncio
import logging
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

from scene import detect_scenes, sample_frames, save_clip
from embeddings import ClipEmbedder, get_or_build_index
from query import expand_query, get_query_embedding
from judge import rank_scenes

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/darky/Documents/video-rag/data/logs/agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VideoSearchAgent:
    
    def __init__(self):
        self.embedder = ClipEmbedder()
        
        Path("/Users/darky/Documents/video-rag/data/faiss").mkdir(parents=True, exist_ok=True)
        Path("/Users/darky/Documents/video-rag/data/output").mkdir(parents=True, exist_ok=True)
        Path("/Users/darky/Documents/video-rag/data/logs").mkdir(parents=True, exist_ok=True)
    
    async def run(self, video_path: str, query: str, top_n: int = 3) -> list[str]:
        video_id = Path(video_path).stem
        print(f"Starting video search for '{query}' in {video_path}")
        
        print("Detecting scenes...")
        scenes = detect_scenes(video_path, scene_duration=5)
        if not scenes:
            print("No scenes detected")
            return []
        
        print(f"Found {len(scenes)} scenes")
        
        print("Sampling frames...")
        scene_frames = sample_frames(video_path, interval=1, timestamps=scenes)
        
        print("Generating embeddings...")
        scene_embeddings = []
        
        for scene_idx in sorted(scene_frames.keys()):
            frames = scene_frames[scene_idx]
            if frames:
                embedding = self.embedder.embed_images(frames)
                scene_embeddings.append(embedding)
            else:
                scene_embeddings.append(np.zeros(self.embedder.embedding_dim))
        
        scene_embeddings = np.array(scene_embeddings)
        
        print("Building search index...")
        index = get_or_build_index(video_id, scene_embeddings, self.embedder.embedding_dim)
        
        print("Processing query...")
        try:
            expanded_query = await expand_query(query)
        except Exception as e:
            print(f"Query expansion failed: {e}. Using original query.")
            expanded_query = query
        
        print("Getting enhanced query embedding...")
        query_embedding = get_query_embedding(expanded_query, self.embedder)
        
        print("Searching for similar scenes...")
        search_k = min(top_n * 2, len(scenes))
        scores, indices = index.search(query_embedding, k=search_k)
        
        if len(indices) == 0 or indices[0] == -1:
            print("No similar scenes found")
            return []
        
        print("Ranking scenes by similarity...")
        candidate_scenes = [str(idx) for idx in indices if idx != -1]
        ranked_scenes = rank_scenes(
            candidate_scenes, 
            expanded_query, 
            scene_embeddings, 
            query_embedding
        )
        
        if not ranked_scenes:
            print("No scenes found with sufficient similarity")
            return []
        
        print("Extracting video clips...")
        output_paths = []
        
        for i, (scene_idx_str, score) in enumerate(ranked_scenes[:top_n]):
            try:
                scene_idx = int(scene_idx_str)
                if scene_idx < len(scenes):
                    start_time, end_time = scenes[scene_idx]
                    
                    output_filename = f"{video_id}_scene_{scene_idx}_score_{score:.3f}.mp4"
                    output_path = f"/Users/darky/Documents/video-rag/data/output/{output_filename}"
                    
                    save_clip(video_path, start_time, end_time, output_path)
                    output_paths.append(output_path)
                    
                    print(f"Extracted clip {i+1}/{top_n}: {output_path} (similarity: {score:.3f})")
                    
            except (ValueError, IndexError) as e:
                print(f"Failed to extract scene {scene_idx_str}: {e}")
                continue
        
        print(f"Search completed. Found {len(output_paths)} clips.")
        return output_paths


async def main():
    video_path = "/Users/darky/Documents/video-rag/data/test_1.mp4"
    query = "man drinking water"
    top_n = 3
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"Searching for '{query}' in {video_path}")
    print(f"Looking for top {top_n} clips (5-7 seconds each)")
    
    agent = VideoSearchAgent()
    clips = await agent.run(video_path, query, top_n)
    
    if clips:
        print(f"Found {len(clips)} relevant clips:")
        for i, clip_path in enumerate(clips, 1):
            print(f"{i}. {clip_path}")
    else:
        print("No relevant clips found.")


if __name__ == "__main__":
    asyncio.run(main())