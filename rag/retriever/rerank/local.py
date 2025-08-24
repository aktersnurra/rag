import os

from loguru import logger as log
from sentence_transformers import CrossEncoder

from rag.message import Messages
from rag.retriever.encoder import Query
from rag.retriever.rerank.abstract import AbstractReranker
from rag.retriever.vector import Documents

Context = Documents | Messages


class Reranker(metaclass=AbstractReranker):
    def __init__(self) -> None:
        self.model = CrossEncoder(os.environ["RERANK_MODEL"], device="cpu")
        self.top_k = int(os.environ["RERANK_TOP_K"])
        self.relevance_threshold = float(os.environ["RERANK_RELEVANCE_THRESHOLD"])

    def rerank(self, query: Query, documents: Context) -> Context:
        results = self.model.rank(
            query=query.query,
            documents=documents.content(),
            return_documents=False,
            top_k=self.top_k,
        )
        rankings = list(
            map(
                lambda x: x.get("corpus_id", 0),
                filter(
                    lambda x: x.get("score", 0.0) > self.relevance_threshold, results
                ),
            )
        )
        log.debug(
            f"Reranking gave {len(rankings)} relevant documents of {len(documents)}"
        )
        documents.rerank(rankings)
        return documents
