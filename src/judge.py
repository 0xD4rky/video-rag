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
