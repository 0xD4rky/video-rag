import os
import pickle
from typing import List, Tuple, Optional
import numpy as np
import torch
import open_clip
import faiss
from PIL import Image
import logging


logging.basicConfig(
    filename="/Users/darky/Documents/video-rag/data/logs/embedding.log",
    format='%(asctime)s %(message)s',
    filemode='w'
    )
logger = logging.getLogger()


class Embeddings():

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai"
    ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def embed_images(
        self,
        frames : List[np.array]
    )-> np.array:
        
        """
        returns: mean-pooled embedding vector
        """

        if not frames:
            return np.zeros(768)  # ViT-L-14 dimension
        
        embeddings = []
        with torch.no_grad():
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)

                image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
        mean_embedding = np.mean(embeddings, axis=0)
        return mean_embedding / np.linalg.norm(mean_embedding)

    def embed_text(
        self,
        text : str
    )-> np.array:
        
        """
        returns: normalized text embedding
        """

        with torch.no_grad():
            text_tokens = self.tokenizer([text]).to(self.device)
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            return text_embeddings.cpu().numpy().flatten()


class Faiss:
    
    def __init__(self, dimension : int = 768, nlist: int = 100):

        """
        dimension: embedding dimension
        nlist: no of clusters
        """

        self.dimension = dimension
        self.nlist = nlist

        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
        self.ids = []

    def add(self, ids: List[int], vectors: np.array) -> None:
        """adding vecs to the index"""

        if not self.index.is_trained and len(vectors) >= self.nlist:
            self.index.train(vectors)        
        if self.index.is_trained:
            self.index.add(vectors)
            self.ids.extend(ids)

    def search(self, query_vector : np.array, k : int = 10)-> Tuple[np.array, np.array]:
        """
        function is used to search for similar vectors from the index
        
        accepts: vector to be searched for and not of similar vectors to be retrieved
        returns: tuple of scores and vectors
        """

        if not self.index.is_trained:
            return np.array([]), np.array([])

        query_vector = query_vector.reshape(1, -1)
        scores, indices = self.index.search(query_vector, k)
        return scores[0], indices[0]

    def save(self, filepath: str) -> None:
        """Save index to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, filepath)
        
        ids_path = filepath.replace('.faiss', '_ids.pkl')
        with open(ids_path, 'wb') as f:
            pickle.dump(self.ids, f)
    
    def load(self, filepath: str) -> None:
        """Load index from disk"""
        self.index = faiss.read_index(filepath)
        
        ids_path = filepath.replace('.faiss', '_ids.pkl')
        if os.path.exists(ids_path):
            with open(ids_path, 'rb') as f:
                self.ids = pickle.load(f)
    

def get_or_build_index(
    video_id: str, 
    scene_vectors: np.ndarray, 
    force: bool = False
) -> Faiss:
    """
    Get existing index or build new one for video.
    
    Args:
        video_id: Unique video identifier
        scene_vectors: Scene embedding vectors
        force: Force rebuild even if index exists
        
    Returns:
        FAISS index ready for search
    """
    index_path = f"data/faiss/{video_id}.faiss"
    
    index = Faiss()
    
    if os.path.exists(index_path) and not force:
        index.load(index_path)
        logger.info(f"Loaded existing index for {video_id}")
    else:
        scene_ids = list(range(len(scene_vectors)))
        index.add(scene_ids, scene_vectors)
        index.save(index_path)
        logger.info(f"Built new index for {video_id} with {len(scene_vectors)} vectors")
    
    return index
    


                


