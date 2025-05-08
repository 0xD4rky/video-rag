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
from tqdm import tqdm

from scene import extract_scenes, save_scene_video

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

load_dotenv()
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

#### video path and query input from the user
video_path = input("Enter the video's path to be searched : ").strip()
text_query = input("Enter your query to be searched in the video : ").strip()


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
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            image_embeddings[url] = image_features.squeeze(0)
            
        except Exception as e:
            print(f"Error processing image from URL {url}: {e}")

    return image_embeddings

def create_text_embeddings(text_query, normalize = True):

    text_inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    # Move inputs to device
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)

    fetched_images = fetch_images_from_google(text_query+"images")
    if fetched_images:
        text_inputs = processor(text=[text_query], return_tensors="pt", padding=True)
        # Move inputs to device
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_embedding = model.get_text_features(**text_inputs)

        images = [img[1] for img in fetched_images]
        image_embeddings = create_image_embeddings(images)
        text_embedding = (text_embedding + image_embeddings.mean(dim = 0, keepdim = True))/2

    return text_embedding

scenes, video_fps = extract_scenes(video_path, scene_duration=5, fps=5)
if not scenes:
    print("No scenes extracted.")
    exit()


def scene_iterations(scenes):
    """
    a function to iterate over each scene separately
    """
    scores = []
    for start_time, end_time, frames, frame_indices in scenes:
        inputs = processor(images = frames, return_tensors = "pt", padding = True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
        scene_embedding = image_embeddings.mean(dim=0, keepdim=True)
        similarity = cosine_similarity(create_text_embeddings(text_query), scene_embedding).item()
        scores.append((start_time, end_time, similarity))

    scores.sort(key = lambda x:x[2], reverse = True)

    return scores

def main():
    
    scene_similarity_scores = []
    scene_similarity_scores = scene_iterations(scenes)

    top_k = 3
    similar_scenes = scene_similarity_scores[:3]
    save_dir = os.path.dirname(video_path) # note: dirname returns the parent directory of the current file
    output_dir = os.path.join(save_dir, "output")
    os.makedirs(output_dir, exist_ok = True)

    for i , (start_time, end_time, similarity) in enumerate(similar_scenes):

        scene_video_path = os.path.join(output_dir, f"scene_{i+1}_{int(start_time)}s_{int(end_time)}s.mp4")
        save_scene_video(video_path, start_time, end_time, scene_video_path)
        print(f"Scene {i+1}: {start_time}s - {end_time}s | Similarity: {similarity:.4f} (Saved: {scene_video_path})")

    print("Top similar scenes retrieved and saved successfully.")


if __name__ == "__main__":

    main()







