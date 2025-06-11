import cv2
from typing import List, Tuple, Dict
import torch
from torch.nn.functional import cosine_similarity
from PIL import Image

from embeddings import load_models, processor, model, device


class VideoJudge:
    """Simple judge that scores scenes using CLIP similarity."""

    def __init__(self):
        load_models()

    @staticmethod
    def extract_key_frames(video_path: str, start_time: float, end_time: float, num_frames: int = 5) -> List[Image.Image]:
        """Extract ``num_frames`` evenly spaced frames from the video segment."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = max(end_frame - start_frame, 1)

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

    def score_scene(self, video_path: str, start_time: float, end_time: float, text_embedding: torch.Tensor) -> float:
        frames = self.extract_key_frames(video_path, start_time, end_time)
        if not frames:
            return -float("inf")

        inputs = processor(images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            img_embeds = model.get_image_features(**inputs)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

        sims = cosine_similarity(text_embedding, img_embeds)
        return sims.max().item()


def select_best_scene(
    video_path: str,
    candidate_scenes: List[Tuple[float, float, float]],
    text_embedding: torch.Tensor,
    alpha: float = 0.6,
) -> Dict:
    """Return the scene with the highest combined similarity score."""

    judge = VideoJudge()
    best = None
    best_score = -float("inf")
    for start, end, base_score in candidate_scenes:
        judge_score = judge.score_scene(video_path, start, end, text_embedding)
        combined = alpha * base_score + (1 - alpha) * judge_score
        if combined > best_score:
            best_score = combined
            best = {
                "start_time": start,
                "end_time": end,
                "combined_score": combined,
            }
    return best