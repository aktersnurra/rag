import os
from abc import abstractmethod
from typing import Type

import cohere
from loguru import logger as log
from sentence_transformers import CrossEncoder

from rag.generator.prompt import Prompt


class AbstractReranker(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @abstractmethod
    def rank(self, prompt: Prompt) -> Prompt:
        return prompt


class Reranker(metaclass=AbstractReranker):
    def __init__(self) -> None:
        self.model = CrossEncoder(os.environ["RERANK_MODEL"])
        self.top_k = int(os.environ["RERANK_TOP_K"])

    def rank(self, prompt: Prompt) -> Prompt:
        if prompt.documents:
            results = self.model.rank(
                query=prompt.query,
                documents=[d.text for d in prompt.documents],
                return_documents=False,
                top_k=self.top_k,
            )
            ranking = list(filter(lambda x: x.get("score", 0.0) > 0.5, results))
            log.debug(
                f"Reranking gave {len(ranking)} relevant documents of {len(prompt.documents)}"
            )
            prompt.documents = [
                prompt.documents[r.get("corpus_id", 0)] for r in ranking
            ]
        return prompt


class CohereReranker(metaclass=AbstractReranker):
    def __init__(self) -> None:
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])
        self.top_k = int(os.environ["RERANK_TOP_K"])

    def rank(self, prompt: Prompt) -> Prompt:
        if prompt.documents:
            response = self.client.rerank(
                model="rerank-english-v3.0",
                query=prompt.query,
                documents=[d.text for d in prompt.documents],
                top_n=self.top_k,
            )
            ranking = list(filter(lambda x: x.relevance_score > 0.5, response.results))
            log.debug(
                f"Reranking gave {len(ranking)} relevant documents of {len(prompt.documents)}"
            )
            prompt.documents = [prompt.documents[r.index] for r in ranking]
        return prompt


def get_reranker(model: str) -> Type[AbstractReranker]:
    match model:
        case "local":
            return Reranker()
        case "cohere":
            return CohereReranker()
        case _:
            exit(1)
