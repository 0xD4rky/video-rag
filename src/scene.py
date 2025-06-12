""" detecting scenes and sampling frames from the videos """

import os
import cv2
import numpy as np
import subprocess
from typing import List, Tuple, Dict
from scenedetect import detect, ContentDetector, ThresholdDetector
import ffmpeg
import logging


logging.basicConfig(
    filename="/Users/darky/Documents/video-rag/data/logs/logs.log",
    format='%(asctime)s %(message)s',
    filemode='w'
    )
logger = logging.getLogger()

def detect_scenes(
        video_path: str,
        method: str = "content",
        min_len_sec: int = 3
)-> List[Tuple[float,float]]:

    """
    dececting scenes using pyscene detect

    returns: List of (start_time, end_time) tuple in seconds
    """

    if method == "content":
        detector = ContentDetector(threshold=30.0, min_scene_len=min_len_sec)
    else:
        detector = ThresholdDetector(threshold=12.0, min_scene_len=min_len_sec)
    
    scene_list = detect(video_path, detector)

    scenes = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        scenes.append((start_time, end_time))
    
    logger.info(f"Detected {len(scenes)} scenes in {video_path}")
    return scenes

def sample_frames(
        video_path : str,
        interval : int,
        timestamps: List[Tuple[float, float]]
)-> Dict[int, List[np.array]]:
    
    """
    scenes -> [frame1, frame2, frame3 .....]
                    |__| -> interval

    returns: dictionary mapping index of frame with its array
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scene_frames = {}
    for scene_idx, (start_time, end_time) in enumerate(timestamps):
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))

        scene_duration_frames = end_frame - start_frame
        if scene_duration_frames >= 3:
            frame_indices = [
                start_frame,
                start_frame + scene_duration_frames // 2,
                end_frame - 1
            ]
        else:
            frame_indices = [start_frame]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                logger.warning(f"Could not read frame {frame_idx}")
        
        scene_frames[scene_idx] = frames

    cap.release()
    logger.info(f"Sampled frames from {len(timestamps)} scenes")
    return scene_frames

def save_clip(
        video_path: str, 
        start: float, 
        end: float, 
        out_path: str
) -> None:
    """
    Save a video clip using ffmpeg-python.
    
    Args:
        video_path: Input video path
        start: Start time in seconds
        end: End time in seconds
        out_path: Output clip path
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    (
        ffmpeg
        .input(video_path, ss=start, t=end-start)
        .output(out_path, vcodec='libx264', acodec='aac')
        .overwrite_output()
        .run(quiet=True)
    )
    
    logger.info(f"Saved clip: {out_path}")

def get_video_info(video_path: str) -> Dict[str, float]:
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

