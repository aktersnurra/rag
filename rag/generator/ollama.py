import os
from typing import Any, Dict, Generator, List

import ollama
from loguru import logger as log

from rag.retriever.vector import Document

from .abstract import AbstractGenerator
from .prompt import ANSWER_INSTRUCTION, Prompt


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
            "Answer the question based only on the following context:\n"
            "<context>\n"
            f"{self.__context(prompt.documents)}\n\n"
            "</context>\n"
            f"{ANSWER_INSTRUCTION}"
            f"Question: {prompt.query.strip()}\n\n"
            "Answer:"
        )
        return metaprompt

    def generate(self, prompt: Prompt) -> Generator[Any, Any, Any]:
        log.debug("Generating answer with ollama...")
        metaprompt = self.__metaprompt(prompt)
        for chunk in ollama.generate(model=self.model, prompt=metaprompt, stream=True):
            yield chunk["response"]

    def chat(self, prompt: Prompt, messages: List[Dict[str, str]]) -> Generator[Any, Any, Any]:
        log.debug("Generating answer with ollama...")
        metaprompt = self.__metaprompt(prompt)
        messages.append({"role": "user", "content": metaprompt})
        for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
            yield chunk["message"]["content"]
