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
import shutil

from scene import extract_scenes, save_scene_video

# Setup device and models lazily so importing this module does not trigger heavy
# initialisation or prompt for user input.  Functions will call ``load_models``
# when needed.

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
model = None
processor = None

load_dotenv()
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

def load_models():
    """Load CLIP model and processor if they are not already initialised."""
    global model, processor
    if model is None or processor is None:
        print(f"Using device: {device}")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def fetch_images_from_google(query, num_images=5, serpapi_key=SERPAPI_KEY):
    search = GoogleSearch({
        "q": query,
        "tbm": "isch",
        "num": num_images,
        "api_key": serpapi_key
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
    load_models()
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

def create_text_embeddings(text_query, normalize=True, use_serpapi=True):
    load_models()
    text_inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)
    
    if normalize:
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    fetched_images = None
    if use_serpapi and SERPAPI_KEY:
        fetched_images = fetch_images_from_google(text_query + " images")
    if fetched_images:
        image_embeddings = create_image_embeddings(fetched_images)
        if image_embeddings is not None:
            # Weighted average between text and image embeddings
            text_embedding = 0.6 * text_embedding + 0.4 * image_embeddings.mean(dim=0, keepdim=True)
            if normalize:
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    return text_embedding

def extract_video_scenes(video_path, scene_duration=5, fps=5):
    """Extract scenes from the given video and report progress."""
    print("Extracting scenes from video...")
    scenes, video_fps = extract_scenes(video_path, scene_duration=scene_duration, fps=fps)
    if not scenes:
        print("No scenes extracted.")
    else:
        print(f"Extracted {len(scenes)} scenes from video")
    return scenes, video_fps

def process_scenes(scenes, text_embedding):
    load_models()
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

def process_video(video_path, text_query, use_serpapi=True):
    print(f"Creating embeddings for query: '{text_query}'")
    text_embedding = create_text_embeddings(text_query, use_serpapi=use_serpapi)

    scenes, _ = extract_video_scenes(video_path, scene_duration=5, fps=5)
    if not scenes:
        return None

    scene_similarity_scores = process_scenes(scenes, text_embedding)

    top_k = min(3, len(scene_similarity_scores))
    similar_scenes = scene_similarity_scores[:top_k]
    
    print(f"\nTop {top_k} scenes from CLIP similarity:")
    for i, (start_time, end_time, similarity) in enumerate(similar_scenes):
        print(f"Scene {i+1}: {start_time:.1f}s - {end_time:.1f}s | Similarity: {similarity:.4f}")
    
    # Use video understanding model to judge the scenes
    print(f"\n{'='*60}")
    print("USING VIDEO UNDERSTANDING MODEL TO SELECT BEST SCENE")
    print(f"{'='*60}")
    
    from judge import select_best_scene
    best_scene = select_best_scene(video_path, similar_scenes, text_embedding)
    
    if best_scene:
        # Save only the best scene selected by the judge
        save_dir = os.path.dirname(video_path) 
        output_dir = os.path.join(save_dir, "output")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        scene_video_path = os.path.join(output_dir, f"best_scene_{int(best_scene['start_time'])}s_{int(best_scene['end_time'])}s.mp4")
        save_scene_video(video_path, best_scene['start_time'], best_scene['end_time'], scene_video_path)
        
        print(f"\n{'='*60}")
        print("FINAL RESULT:")
        print(f"{'='*60}")
        print(f"Best scene saved: {os.path.basename(scene_video_path)}")
        print(f"Time span: {best_scene['start_time']:.1f}s - {best_scene['end_time']:.1f}s")
        print(f"Final score: {best_scene['combined_score']:.2f}")
        print(f"Location: {scene_video_path}")
        print("Scene retrieved and saved successfully!")
    else:
        print("Could not select the best scene.")

    return best_scene

def main():
    video_path = input("Enter the video's path to be searched : ").strip()
    text_query = input("Enter your query to be searched in the video : ").strip()
    process_video(video_path, text_query)


if __name__ == "__main__":
    main()
