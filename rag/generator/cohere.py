import os
from dataclasses import asdict
from typing import Any, Generator, List

import cohere
from loguru import logger as log

from rag.message import Message
from rag.retriever.vector import Document

from .abstract import AbstractGenerator


class Cohere(metaclass=AbstractGenerator):
    def __init__(self) -> None:
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])

    def generate(
        self, messages: List[Message], documents: List[Document]
    ) -> Generator[Any, Any, Any]:
        log.debug("Generating answer from cohere...")
        for event in self.client.chat_stream(
            message=messages[-1].content,
            documents=[asdict(d) for d in documents],
            chat_history=[m.as_dict() for m in messages[:-1]],
            prompt_truncation="AUTO",
        ):
            if event.event_type == "text-generation":
                yield event.text
            elif event.event_type == "citation-generation":
                yield event.citations
            elif event.event_type == "stream-end":
                yield event.finish_reason
