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

#### taking the query input from the user
text_query = input("Enter your query to be searched in the video").strip()


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


def create_image_embeddings(images, normalize=True):
    """
    Creates embeddings for a list of images using the CLIP model.
    
    The function processes images retrieved from SERPAPI and generates
    vector representations (embeddings) that capture the semantic content
    of each image. These embeddings can be used for similarity comparison,
    search, and other computer vision tasks.
    
    Args:
        images: List of tuples containing (url, PIL.Image) from fetch_images_from_google
        normalize: Whether to normalize embeddings (recommended for similarity comparison)
    
    Returns:
        Dictionary mapping image URLs to their corresponding embeddings
    """
    image_embeddings = {}
    
    for url, img in images:
        try:
            inputs = processor(images=img, return_tensors="pt", padding=True)

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            image_embeddings[url] = image_features.squeeze(0)
            
        except Exception as e:
            print(f"Error processing image from URL {url}: {e}")
    
    return image_embeddings

def create_text_embeddings(text, normalize = True):

    text_inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)

    fetched_images = fetch_images_from_google(text_query+"images")
    if fetched_images:
        images = [img[1] for img in fetched_images]

    #text_embedding = text_embedding + create_image_embeddings()