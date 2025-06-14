import os
import asyncio
from typing import List
import numpy as np
import requests
from PIL import Image
import google.generativeai as genai
from serpapi import GoogleSearch
from tenacity import retry, stop_after_attempt, wait_exponential
from embeddings import ClipEmbedder
import logging
from io import BytesIO

logging.basicConfig(
    filename="/Users/darky/Documents/video-rag/data/logs/query.log",
    format='%(asctime)s %(message)s',
    filemode='w'
    )
logger = logging.getLogger()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def expand_query(
        query: str, 
        temperature: float = 0.7
    )-> str:

    """
    Expand query using Gemini for better search results
    """

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    Expand this video search query to include related visual concepts, actions, objects, and scenarios.
    Keep it concise but comprehensive for better video matching.
    
    Original query: {query}
    
    Expanded query:
    """

    response = await asyncio.to_thread(
        model.generate_content, 
        prompt, 
        generation_config=genai.types.GenerationConfig(temperature=temperature)
    )
    expanded = response.text.strip()
    logger.info(f"Expanded query: '{query}' -> '{expanded}'")
    return expanded


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_serp_images(query: str, k: int = 5) -> List[np.ndarray]:
    """
    Fetch reference images from Google Images via SerpAPI.
    
    Args:
        query: Search query
        k: Number of images to fetch
        
    Returns:
        List of image arrays
    """
    search = GoogleSearch({
        "q": query,
        "tbm": "isch",
        "api_key": os.getenv("SERP_API_KEY"),
        "num": k
    })
    
    results = search.get_dict()
    images = []
    
    if "images_results" in results:
        for img_result in results["images_results"][:k]:
            try:
                img_url = img_result["original"]
                response = requests.get(img_url, timeout=10)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content))
                image = image.convert("RGB")
                image_array = np.array(image)
                images.append(image_array)
                
            except Exception as e:
                logger.warning(f"Failed to fetch image: {e}")
                continue
    
    logger.info(f"Fetched {len(images)} reference images for query: {query}")
    return images


def get_query_embedding(expanded_query: str, embedder: ClipEmbedder) -> np.ndarray:
    """
    generating embedding for expanded query.
    
    returns: query embedidng vector
    """
    return embedder.embed_text(expanded_query)