import os
from typing import List

from loguru import logger as log
from sentence_transformers import CrossEncoder

from rag.message import Message
from rag.retriever.rerank.abstract import AbstractReranker
from rag.retriever.vector import Document


class Reranker(metaclass=AbstractReranker):
    def __init__(self) -> None:
        self.model = CrossEncoder(os.environ["RERANK_MODEL"], device="cpu")
        self.top_k = int(os.environ["RERANK_TOP_K"])
        self.relevance_threshold = float(os.environ["RETRIEVER_RELEVANCE_THRESHOLD"])

    def rerank_documents(self, query: str, documents: List[Document]) -> List[str]:
        results = self.model.rank(
            query=query,
            documents=[d.text for d in documents],
            return_documents=False,
            top_k=self.top_k,
        )
        ranking = list(
            filter(lambda x: x.get("score", 0.0) > self.relevance_threshold, results)
        )
        log.debug(
            f"Reranking gave {len(ranking)} relevant documents of {len(documents)}"
        )
        return [documents[r.get("corpus_id", 0)] for r in ranking]

    def rerank_messages(self, query: str, messages: List[Message]) -> List[Message]:
        results = self.model.rank(
            query=query,
            documents=[m.content for m in messages],
            return_documents=False,
            top_k=self.top_k,
        )
        ranking = list(
            filter(lambda x: x.get("score", 0.0) > self.relevance_threshold, results)
        )
        log.debug(
            f"Reranking gave {len(ranking)} relevant chat messages of {len(messages)}"
        )
        return [messages[r.get("corpus_id", 0)] for r in ranking]
