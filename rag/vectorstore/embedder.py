"""Embedding model wrapper"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm


class Embedder:
    """Generate embeddings for text"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query"""
        return self.model.encode(query, convert_to_numpy=True)
