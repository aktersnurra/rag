import os
from typing import Any, Generator, List

import ollama
from loguru import logger as log

from rag.retriever.vector import Document

from .abstract import AbstractGenerator
from .prompt import Prompt


class Ollama(metaclass=AbstractGenerator):
    def __init__(self) -> None:
        self.model = os.environ["GENERATOR_MODEL"]

    def __context(self, documents: List[Document]) -> str:
        results = [
            f"Document: {i}\ntitle: {doc.title}\ntext: {doc.text}"
            for i, doc in enumerate(documents)
        ]
        return "\n".join(results)

    def __metaprompt(self, prompt: Prompt) -> str:
        metaprompt = (
            "Answer the question based only on the following context:"
            "<context>\n"
            f"{self.__context(prompt.documents)}\n\n"
            "</context>\n"
            "Given the context information and not prior knowledge, answer the question."
            "If the context is irrelevant to the question, answer that you do not know "
            "the answer to the question given the context.\n"
            f"Question: {prompt.query.strip()}\n\n"
            "Answer:"
        )
        return metaprompt

    def generate(self, prompt: Prompt) -> Generator[Any, Any, Any]:
        log.debug("Generating answer with ollama...")
        metaprompt = self.__metaprompt(prompt)
        for chunk in ollama.generate(model=self.model, prompt=metaprompt, stream=True):
            yield chunk["response"]
