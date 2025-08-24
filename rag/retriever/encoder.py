from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Union

import ollama
from langchain_core.documents import Document
from loguru import logger as log
from qdrant_client.http.models import StrictFloat
from tqdm import tqdm

from .vector import Documents, Point

@dataclass
class Query:
    query: str


Input = Query | Documents


class Encoder:
    def __init__(self) -> None:
        self.model = os.environ["ENCODER_MODEL"]
        self.preamble = (
            "Represent this sentence for searching relevant passages: "
            if "mxbai-embed-large" in model_name
            else ""
        )

    def __get_source(self, metadata: Dict[str, str]) -> str:
        source = metadata["source"]
        return Path(source).name

    def __encode(self, prompt: str) -> List[StrictFloat]:
        return list(ollama.embeddings(model=self.model, prompt=prompt)["embedding"])

    # TODO: move this to vec db and just return the embeddings
    # TODO: use late chunking here
    def __encode_document(self, chunks: List[Document]) -> List[Point]:
        log.debug("Encoding document...")
        return [
            Point(
                id=hashlib.sha256(
                    chunk.page_content.encode(encoding="utf-8")
                ).hexdigest(),
                vector=list(self.__encode(chunk.page_content)),
                payload={
                    "text": chunk.page_content,
                    "source": self.__get_source(chunk.metadata),
                },
            )
            for chunk in tqdm(chunks)
        ]

    def __encode_query(self, query: str) -> List[StrictFloat]:
        log.debug(f"Encoding query: {query}")
        query = self.preamble + query
        return self.__encode(query)

    def encode(self, x: Input) -> Union[List[StrictFloat], List[Point]]:
        match x:
            case Query(query):
                return self.__encode_query(query)
            case Documents(documents):
                return self.__encode_document(documents)
