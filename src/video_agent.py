import os
from typing import Optional

from query_expander import QueryExpander
from scene import extract_scenes, save_scene_video
from embeddings import create_text_embeddings, process_scenes
from judge import select_best_scene


class VideoAgent:
    """High level agent orchestrating the video retrieval workflow."""

    def __init__(self, serpapi_key: Optional[str] = None, gemini_key: Optional[str] = None, use_serpapi: bool = True):
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")
        self.use_serpapi = use_serpapi and bool(self.serpapi_key)
        self.expander = QueryExpander(gemini_key)

    def run(self, video_path: str, query: str, output_dir: Optional[str] = None) -> Optional[str]:
        expanded = self.expander.expand(query)

        text_embedding = create_text_embeddings(expanded, use_serpapi=self.use_serpapi)

        scenes, _ = extract_scenes(video_path, scene_duration=5, fps=5)
        if not scenes:
            return None

        scene_scores = process_scenes(scenes, text_embedding)
        top_k = min(3, len(scene_scores))
        candidates = scene_scores[:top_k]

        best_scene = select_best_scene(video_path, candidates, text_embedding)
        if not best_scene:
            return None

        output_dir = output_dir or os.path.join(os.path.dirname(video_path), "output")
        os.makedirs(output_dir, exist_ok=True)
        scene_video_path = os.path.join(
            output_dir,
            f"best_scene_{int(best_scene['start_time'])}s_{int(best_scene['end_time'])}s.mp4",
        )
        save_scene_video(video_path, best_scene['start_time'], best_scene['end_time'], scene_video_path)
        return scene_video_path

