import os
from typing import List

import numpy as np
import ollama


class Encoder:
    def __init__(self) -> None:
        self.model = os.environ["ENCODER_MODEL"]
        self.query_prompt = "Represent this sentence for searching relevant passages: "

    def __encode(self, prompt: str) -> np.ndarray:
        x = ollama.embeddings(model=ENCODER_MODEL, prompt=prompt)
        x = np.array([x["embedding"]]).astype("float32")
        return x

    def encode(self, doc: List[str]) -> List[np.ndarray]:
        return [self.__encode(chunk) for chunk in doc]

    def query(self, query: str) -> np.ndarray:
        query = self.query_prompt + query
        return self.__encode(query)
