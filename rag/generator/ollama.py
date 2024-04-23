import os
from typing import Any, Generator, List

import ollama
from loguru import logger as log

from rag.retriever.vector import Document

from .abstract import AbstractGenerator
from .prompt import ANSWER_INSTRUCTION, Prompt


class Ollama(metaclass=AbstractGenerator):
    def __init__(self) -> None:
        self.model = os.environ["GENERATOR_MODEL"]
        log.debug(f"Using {self.model} for generator...")

    def __context(self, documents: List[Document]) -> str:
        results = [
            f"Document: {i}\ntitle: {doc.title}\ntext: {doc.text}"
            for i, doc in enumerate(documents)
        ]
        return "\n".join(results)

    def __metaprompt(self, prompt: Prompt) -> str:
        metaprompt = (
            "Context information is below.\n"
            "---------------------\n"
            f"{self.__context(prompt.documents)}\n\n"
            "---------------------\n"
            f"{ANSWER_INSTRUCTION}"
            "Do not attempt to answer the query without relevant context and do not use"
            " prior knowledge or training data!\n"
            f"Query: {prompt.query.strip()}\n\n"
            "Answer:"
        )
        return metaprompt

    def generate(self, prompt: Prompt) -> Generator[Any, Any, Any]:
        log.debug("Generating answer with ollama...")
        metaprompt = self.__metaprompt(prompt)
        for chunk in ollama.generate(model=self.model, prompt=metaprompt, stream=True):
            yield chunk["response"]
