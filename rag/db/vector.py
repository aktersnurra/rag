from typing import Tuple
import faiss
import numpy as np

# TODO: read from .env
EMBEDDING_DIM = 1024


# TODO: inner product distance metric?
class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        # TODO: load from file

    def add(self, embeddings: np.ndarray):
        # TODO: save to file
        self.index.add(embeddings)

    def search(
        self, query: np.ndarray, neighbors: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        score, indices = self.index.search(query, neighbors)
        return score, indices
