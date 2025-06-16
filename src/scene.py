import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
from scenedetect import detect, ContentDetector, ThresholdDetector
import ffmpeg
import logging

logger = logging.getLogger(__name__)


def detect_scenes(
    video_path: str,
    method: str = "content", 
    min_len_sec: int = 2,
    scene_duration: int = 5
) -> List[Tuple[float, float]]:
    
    """
    detecting scenes with duration (5-7 seconds).
    
    args:
        video_path: Path to video file
        method: Detection method ('content' or 'threshold')  
        min_len_sec: Minimum scene length in seconds
        scene_duration: Target scene duration in seconds
        
    returns:
        List of (start_time, end_time) tuples in seconds
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / video_fps
    cap.release()
    
    scenes = []
    for start_time in range(0, int(duration), scene_duration):
        end_time = min(start_time + scene_duration, duration)
        
        if end_time - start_time >= 5:
            actual_duration = min(7, max(5, end_time - start_time))
            end_time = start_time + actual_duration
        
        scenes.append((float(start_time), float(end_time)))
    
    logger.info(f"Created {len(scenes)} smart scenes in {video_path}")
    return scenes


def sample_frames(
    video_path: str,
    interval: int,
    timestamps: List[Tuple[float, float]],
    fps: int = 5
) -> Dict[int, List[np.ndarray]]:
    
    """
    sample frames from video scenes with better distribution
        
    returns:
        Dictionary mapping scene index to list of frame arrays
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    scene_frames = {}
    
    for scene_idx, (start_time, end_time) in enumerate(timestamps):
        frames = []
        frame_indices = []
        
        for t in range(int(start_time * video_fps), int(end_time * video_fps), video_fps // fps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(t)
        
        if not frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * video_fps))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        scene_frames[scene_idx] = frames
    
    cap.release()
    logger.info(f"Sampled frames from {len(timestamps)} scenes")
    return scene_frames


def save_clip(video_path: str, start: float, end: float, out_path: str) -> None:
    """
    save a video clip using opencv with exact duration control.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    while cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    
    duration = end - start
    logger.info(f"Saved clip: {out_path} (duration: {duration:.2f}s)")


def get_video_info(video_path: str) -> Dict[str, float]:
    """
    get video metadata information.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': cap.get(cv2.CAP_PROP_FRAME_COUNT),
        'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info