import os
import pickle
from typing import List, Optional
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import logging

logger = logging.getLogger(__name__)


class ClipEmbedder:
    """CLIP-based image and text embedder"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            dummy_text = self.processor(text=["test"], return_tensors="pt", padding=True)
            dummy_text = {k: v.to(self.device) for k, v in dummy_text.items()}
            dummy_embedding = self.model.get_text_features(**dummy_text)
            self.embedding_dim = dummy_embedding.shape[1]

    def embed_images(self, frames: List[np.ndarray]) -> np.ndarray:
        """for mean pooling image embeddings of the sampled scenes"""
        if not frames:
            return np.zeros(self.embedding_dim)
        
        pil_images = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            pil_images.append(pil_image)
        
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
            #mean pool and normalize
            scene_embedding = image_embeddings.mean(dim=0, keepdim=True)
            scene_embedding = scene_embedding / scene_embedding.norm(dim=-1, keepdim=True)
            
        return scene_embedding.cpu().numpy().flatten()

    def embed_text(self, text: str) -> np.ndarray:
        """normalized text embeddings"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            
        return text_embedding.cpu().numpy().flatten()

    def embed_image_list(self, images: List[Image.Image]) -> np.ndarray:
        """Embed a list of PIL images."""
        if not images:
            return np.zeros(self.embedding_dim)
            
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
            #mean pool and normalize
            mean_embedding = image_embeddings.mean(dim=0, keepdim=True)
            mean_embedding = mean_embedding / mean_embedding.norm(dim=-1, keepdim=True)
            
        return mean_embedding.cpu().numpy().flatten()


class FaissIndex:
    """FAISS-based similarity search index."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  
        self.ids = []

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        """Add vectors to the index."""
        if vectors.size == 0:
            return
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        self.index.add(vectors.astype(np.float32))
        self.ids.extend(ids)

    def search(self, query_vector: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        if query_vector.size == 0:
            return np.array([]), np.array([])
        
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        k = min(k, len(self.ids)) if self.ids else k
        
        if k <= 0:
            return np.array([]), np.array([])
        
        scores, indices = self.index.search(query_vector, k)
        return scores[0], indices[0]

    def save(self, filepath: str) -> None:

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, filepath)
        
        ids_path = filepath.replace('.faiss', '_ids.pkl')
        with open(ids_path, 'wb') as f:
            pickle.dump(self.ids, f)

    def load(self, filepath: str) -> None:

        self.index = faiss.read_index(filepath)
        
        ids_path = filepath.replace('.faiss', '_ids.pkl')
        if os.path.exists(ids_path):
            with open(ids_path, 'rb') as f:
                self.ids = pickle.load(f)


def get_or_build_index(
    video_id: str, 
    scene_vectors: np.ndarray, 
    dimension: int,
    force: bool = False
) -> FaissIndex:
    
    """Get existing index or build new one for video"""
    
    index_path = f"/Users/darky/Documents/video-rag/data/faiss/{video_id}.faiss"
    
    index = FaissIndex(dimension)
    
    if os.path.exists(index_path) and not force:
        index.load(index_path)
        logger.info(f"Loaded existing index for {video_id}")
    else:
        scene_ids = list(range(len(scene_vectors)))
        index.add(scene_ids, scene_vectors)
        index.save(index_path)
        logger.info(f"Built new index for {video_id} with {len(scene_vectors)} vectors")
    
    return index