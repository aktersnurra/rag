import os

from loguru import logger as log
from sentence_transformers import CrossEncoder

from rag.generator.prompt import Prompt
from rag.retriever.rerank.abstract import AbstractReranker


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
