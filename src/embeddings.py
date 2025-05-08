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
import numpy as np

from scene import extract_scenes, save_scene_video

# Use CUDA if available, otherwise MPS, then CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load models
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

load_dotenv()
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

# Input from the user
video_path = input("Enter the video's path to be searched : ").strip()
text_query = input("Enter your query to be searched in the video : ").strip()


def fetch_images_from_google(query, num_images=5):
    search = GoogleSearch({
        "q": query,
        "tbm": "isch",
        "num": num_images,
        "api_key": SERPAPI_KEY
    })
    results = search.get_dict()
    image_urls = [img["original"] for img in results.get("images_results", [])[:num_images]]
    print(f"Fetching {len(image_urls)} images for query: '{query}'")
    
    images = []
    for url in tqdm(image_urls, desc="Downloading images"):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append((url, img))
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
    return images

def create_image_embeddings(images, normalize=True):
    image_embeddings = []
    
    for url, img in tqdm(images, desc="Creating image embeddings"):
        try:
            inputs = processor(images=img, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            image_embeddings.append(image_features.squeeze(0))
            
        except Exception as e:
            print(f"Error processing image: {e}")

    if image_embeddings:
        return torch.stack(image_embeddings)
    return None

def create_text_embeddings(text_query, normalize=True):
    text_inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)
    
    if normalize:
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    fetched_images = fetch_images_from_google(text_query + " images")
    if fetched_images:
        image_embeddings = create_image_embeddings(fetched_images)
        if image_embeddings is not None:
            # Weighted average between text and image embeddings
            text_embedding = 0.6 * text_embedding + 0.4 * image_embeddings.mean(dim=0, keepdim=True)
            if normalize:
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    return text_embedding

print("Extracting scenes from video...")
scenes, video_fps = extract_scenes(video_path, scene_duration=5, fps=5)
if not scenes:
    print("No scenes extracted.")
    exit()
print(f"Extracted {len(scenes)} scenes from video")

def process_scenes(scenes, text_embedding):
    scores = []
    text_embedding = text_embedding.to(device)
    
    for start_time, end_time, frames, frame_indices in tqdm(scenes, desc="Processing scenes"):
        inputs = processor(images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        
        # Calculate similarity for each frame in the scene
        similarities = cosine_similarity(text_embedding, image_embeddings)
        
        # Use both mean and max for better results
        mean_similarity = similarities.mean().item()
        max_similarity = similarities.max().item()
        
        # Combine scores with weighted average (favoring max a bit more)
        combined_score = 0.4 * mean_similarity + 0.6 * max_similarity
        scores.append((start_time, end_time, combined_score))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores

def main():
    print(f"Creating embeddings for query: '{text_query}'")
    text_embedding = create_text_embeddings(text_query)
    
    scene_similarity_scores = process_scenes(scenes, text_embedding)

    top_k = min(3, len(scene_similarity_scores))
    similar_scenes = scene_similarity_scores[:top_k]
    
    save_dir = os.path.dirname(video_path) 
    output_dir = os.path.join(save_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving top {top_k} scenes:")
    for i, (start_time, end_time, similarity) in enumerate(similar_scenes):
        scene_video_path = os.path.join(output_dir, f"scene_{i+1}_{int(start_time)}s_{int(end_time)}s.mp4")
        save_scene_video(video_path, start_time, end_time, scene_video_path)
        print(f"Scene {i+1}: {start_time:.1f}s - {end_time:.1f}s | Similarity: {similarity:.4f} (Saved: {os.path.basename(scene_video_path)})")

    print("\nTop similar scenes retrieved and saved successfully.")


if __name__ == "__main__":
    main()







