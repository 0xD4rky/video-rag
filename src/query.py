import os
import asyncio
from typing import List
import numpy as np
import requests
from io import BytesIO
import google.generativeai as genai
from serpapi import GoogleSearch
from PIL import Image
from embeddings import ClipEmbedder
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def expand_query(query: str, temperature: float = 0.7) -> str:
    """Expand query using Gemini for better search results.
    
    Args:
        query: Original search query
        temperature: LLM sampling temperature
        
    Returns:
        Expanded query string
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_key:
        logger.warning("No GEMINI_API_KEY found for query expansion")
        return query
    
    expansion_prompt = f"""
    Expand this video search query to include related visual elements, actions, objects, and contexts that would help find relevant video scenes. Keep it concise but comprehensive.
    
    Original query: {query}
    
    Expanded query:"""
    
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-pro')
        response = await asyncio.to_thread(
            model.generate_content, 
            expansion_prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        expanded = response.text.strip()
        logger.info(f"Expanded query: '{query}' -> '{expanded}'")
        return expanded
        
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return query


def fetch_images_from_google(query: str, num_images: int = 2) -> List[Image.Image]:
    """Fetch images from Google using SerpAPI.
    
    Args:
        query: Search query
        num_images: Number of images to fetch
        
    Returns:
        List of PIL Images
    """
    serpapi_key = os.getenv("SERPAPI_KEY")
    
    if not serpapi_key:
        logger.warning("No SERPAPI_KEY found for image fetching")
        return []
    
    try:
        search = GoogleSearch({
            "q": query + " image",
            "tbm": "isch",
            "num": num_images,
            "api_key": serpapi_key
        })
        results = search.get_dict()
        image_urls = [img["original"] for img in results.get("images_results", [])[:num_images]]
        logger.info(f"Fetched {len(image_urls)} image URLs for query: {query}")
        
        images = []
        for url in image_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    images.append(img)
            except Exception as e:
                logger.warning(f"Failed to fetch image from {url}: {e}")
                continue
        
        return images
        
    except Exception as e:
        logger.warning(f"Image fetching failed: {e}")
        return []


def get_enhanced_query_embedding(expanded_query: str, embedder: ClipEmbedder) -> np.ndarray:
    """Generate enhanced embedding combining text and reference images.
    
    Args:
        expanded_query: Expanded search query
        embedder: CLIP embedder instance
        
    Returns:
        Enhanced query embedding vector
    """

    text_embedding = embedder.embed_text(expanded_query)
    fetched_images = fetch_images_from_google(expanded_query, num_images=2)
    
    if fetched_images:

        image_embedding = embedder.embed_image_list(fetched_images)
        enhanced_embedding = (text_embedding + image_embedding) / 2 # text + image embeddings (shared space)
        logger.info(f"Enhanced query embedding with {len(fetched_images)} reference images")
        return enhanced_embedding
    else:
        logger.info("Using text-only embedding")
        return text_embedding


def get_query_embedding(expanded_query: str, embedder: ClipEmbedder) -> np.ndarray:
    """Generate embedding for expanded query.
    
    Args:
        expanded_query: Expanded search query
        embedder: CLIP embedder instance
        
    Returns:
        Query embedding vector
    """
    return get_enhanced_query_embedding(expanded_query, embedder)