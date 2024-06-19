import os
from typing import Any, Generator, List

import ollama
from loguru import logger as log

from rag.message import Message
from rag.retriever.vector import Document

from .abstract import AbstractGenerator


class Ollama(metaclass=AbstractGenerator):
    def __init__(self) -> None:
        self.model = os.environ["GENERATOR_MODEL"]
        log.debug(f"Using {self.model} for generator...")

    def generate(
        self, messages: List[Message], documents: List[Document]
    ) -> Generator[Any, Any, Any]:
        log.debug("Generating answer with ollama...")
        messages = [m.as_dict() for m in messages]
        for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
            yield chunk["message"]["content"]
