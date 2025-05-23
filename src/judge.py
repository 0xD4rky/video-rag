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
