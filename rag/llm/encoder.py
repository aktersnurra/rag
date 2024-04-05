import os
from typing import List, Optional
from uuid import uuid4

import ollama
from langchain_core.documents import Document
from loguru import logger as log
from qdrant_client.http.models import StrictFloat

from rag.db.vector import Point


class Encoder:
    def __init__(self) -> None:
        self.model = os.environ["ENCODER_MODEL"]
        self.query_prompt = "Represent this sentence for searching relevant passages: "

    def __encode(self, prompt: str) -> List[StrictFloat]:
        return list(ollama.embeddings(model=self.model, prompt=prompt)["embedding"])

    def encode_document(self, chunks: List[Document]) -> List[Point]:
        log.debug("Encoding document...")
        return [
            Point(
                id=str(uuid4()),
                vector=self.__encode(chunk.page_content),
                payload={"text": chunk.page_content},
            )
            for chunk in chunks
        ]

    def encode_query(self, query: str) -> List[StrictFloat]:
        log.debug(f"Encoding query: {query}")
        query = self.query_prompt + query
        return self.__encode(query)
