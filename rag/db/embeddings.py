import os
from typing import Tuple

import faiss
import numpy as np


# TODO: inner product distance metric?
class Embeddings:
    def __init__(self):
        self.dim = int(os.environ["EMBEDDING_DIM"])
        self.index = faiss.IndexFlatL2(self.dim)
        # TODO: load from file

    def add(self, embeddings: np.ndarray):
        # TODO: save to file
        self.index.add(embeddings)

    def search(
        self, query: np.ndarray, neighbors: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        score, indices = self.index.search(query, neighbors)
        return score, indices
