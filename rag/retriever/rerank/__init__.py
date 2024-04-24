from typing import Type

from rag.retriever.rerank.abstract import AbstractReranker
from rag.retriever.rerank.cohere import CohereReranker
from rag.retriever.rerank.local import Reranker


def get_reranker(model: str) -> Type[AbstractReranker]:
    match model:
        case "local":
            return Reranker()
        case "cohere":
            return CohereReranker()
        case _:
            exit(1)
