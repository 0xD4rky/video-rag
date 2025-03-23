import os
import torch
import cv2
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch.nn.functional import cosine_similarity
from serpapi import GoogleSearch
from dotenv import load_dotenv

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

load_dotenv()
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")


def fetch_images_from_google(query, num_images = 3):

    """
    ROLE: 

        This function is mainly used to search images from google using the SERPAPI based on the query provided by the user

    ARGS:

        query -> query given by the user so search the video
        num_images -> the number of images to be searched and retrieved from google

    RETURNS:

        a list of tuple containing urls and the pixel arrays of the retrieved images
    """

    search = GoogleSearch({
        "q": query,
        "tbm": "isch",
        "num": num_images,
        "api_key": SERPAPI_KEY
    })
    results = search.get_dict()
    image_urls = [img["original"] for img in results.get("images_results", [])[:num_images]]
    print("Fetched image URLs:", image_urls)
    
    images = []
    for url in image_urls:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append((url, img))
    return images
