import hashlib
import os
from pathlib import Path
from typing import Dict, List

import ollama
from langchain_core.documents import Document
from loguru import logger as log
from qdrant_client.http.models import StrictFloat
from tqdm import tqdm

from .vector import Point


class Encoder:
    def __init__(self) -> None:
        self.model = os.environ["ENCODER_MODEL"]
        self.query_prompt = "Represent this sentence for searching relevant passages: "

    def __encode(self, prompt: str) -> List[StrictFloat]:
        return list(ollama.embeddings(model=self.model, prompt=prompt)["embedding"])

    def __get_source(self, metadata: Dict[str, str]) -> str:
        source = metadata["source"]
        return Path(source).name

    def encode_document(self, chunks: List[Document]) -> List[Point]:
        log.debug("Encoding document...")
        return [
            Point(
                id=hashlib.sha256(
                    chunk.page_content.encode(encoding="utf-8")
                ).hexdigest(),
                vector=self.__encode(chunk.page_content),
                payload={
                    "text": chunk.page_content,
                    "source": self.__get_source(chunk.metadata),
                },
            )
            for chunk in tqdm(chunks)
        ]

    def encode_query(self, query: str) -> List[StrictFloat]:
        log.debug(f"Encoding query: {query}")
        if self.model == "mxbai-embed-large":
            query = self.query_prompt + query
        return self.__encode(query)
