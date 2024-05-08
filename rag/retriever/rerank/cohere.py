import os

import cohere
from loguru import logger as log

from rag.generator.prompt import Prompt
from rag.retriever.rerank.abstract import AbstractReranker


class CohereReranker(metaclass=AbstractReranker):
    def __init__(self) -> None:
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])
        self.top_k = int(os.environ["RERANK_TOP_K"])
        self.relevance_threshold = float(os.environ["RETRIEVER_RELEVANCE_THRESHOLD"])

    def rank(self, prompt: Prompt) -> Prompt:
        if prompt.documents:
            response = self.client.rerank(
                model="rerank-english-v3.0",
                query=prompt.query,
                documents=[d.text for d in prompt.documents],
                top_n=self.top_k,
            )
            ranking = list(
                filter(
                    lambda x: x.relevance_score > self.relevance_threshold,
                    response.results,
                )
            )
            log.debug(
                f"Reranking gave {len(ranking)} relevant documents of {len(prompt.documents)}"
            )
            prompt.documents = [prompt.documents[r.index] for r in ranking]
        return prompt
