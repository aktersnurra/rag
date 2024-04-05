import os
from typing import List

import numpy as np
import ollama
from langchain_core.documents import Document


class Encoder:
    def __init__(self) -> None:
        self.model = os.environ["ENCODER_MODEL"]
        self.query_prompt = "Represent this sentence for searching relevant passages: "

    def __encode(self, prompt: str) -> np.ndarray:
        x = ollama.embeddings(model=self.model, prompt=prompt)
        x = np.array([x["embedding"]]).astype("float32")
        return x

    def encode_document(self, chunks: List[Document]) -> np.ndarray:
        return np.concatenate([self.__encode(chunk.page_content) for chunk in chunks])

    def query(self, query: str) -> np.ndarray:
        query = self.query_prompt + query
        return self.__encode(query)
