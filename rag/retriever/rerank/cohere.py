import os
from typing import List

import cohere
from loguru import logger as log

from rag.rag import Message
from rag.retriever.rerank.abstract import AbstractReranker
from rag.retriever.vector import Document


class CohereReranker(metaclass=AbstractReranker):
    def __init__(self) -> None:
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])
        self.top_k = int(os.environ["RERANK_TOP_K"])
        self.relevance_threshold = float(os.environ["RETRIEVER_RELEVANCE_THRESHOLD"])

    def rerank_documents(self, query: str, documents: List[Document]) -> List[str]:
        response = self.client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[d.text for d in documents],
            top_n=self.top_k,
        )
        ranking = list(
            filter(
                lambda x: x.relevance_score > self.relevance_threshold,
                response.results,
            )
        )
        log.debug(
            f"Reranking gave {len(ranking)} relevant documents of {len(documents)}"
        )
        return [documents[r.index] for r in ranking]


    def rerank_messages(self, query: str, messages: List[Message]) -> List[Message]:
        response = self.model.rank(
            query=query,
            documents=[m.message for m in messages],
            return_documents=False,
            top_k=self.top_k,
        )
        ranking = list(
            filter(
                lambda x: x.relevance_score > self.relevance_threshold,
                response.results,
            )
        )
        log.debug(
            f"Reranking gave {len(ranking)} relevant chat messages of {len(messages)}"
        )
        return [messages[r.index] for r in ranking]
