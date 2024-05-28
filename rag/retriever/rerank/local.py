import os
from typing import List

from loguru import logger as log
from sentence_transformers import CrossEncoder

from rag.generator.prompt import Prompt
from rag.retriever.memory import Log
from rag.retriever.rerank.abstract import AbstractReranker


class Reranker(metaclass=AbstractReranker):
    def __init__(self) -> None:
        self.model = CrossEncoder(os.environ["RERANK_MODEL"])
        self.top_k = int(os.environ["RERANK_TOP_K"])
        self.relevance_threshold = float(os.environ["RETRIEVER_RELEVANCE_THRESHOLD"])

    def rank(self, prompt: Prompt) -> Prompt:
        if prompt.documents:
            results = self.model.rank(
                query=prompt.query,
                documents=[d.text for d in prompt.documents],
                return_documents=False,
                top_k=self.top_k,
            )
            ranking = list(
                filter(
                    lambda x: x.get("score", 0.0) > self.relevance_threshold, results
                )
            )
            log.debug(
                f"Reranking gave {len(ranking)} relevant documents of {len(prompt.documents)}"
            )
            prompt.documents = [
                prompt.documents[r.get("corpus_id", 0)] for r in ranking
            ]
        return prompt

    def rank_memory(self, prompt: Prompt, history: List[Log]) -> List[Log]:
        if history:
            results = self.model.rank(
                query=prompt.query,
                documents=[m.bot.message for m in history],
                return_documents=False,
                top_k=self.top_k,
            )
            ranking = list(
                filter(
                    lambda x: x.get("score", 0.0) > self.relevance_threshold, results
                )
            )
            log.debug(
                f"Reranking gave {len(ranking)} relevant messages of {len(history)}"
            )
            history = [history[r.get("corpus_id", 0)] for r in ranking]
        return history
