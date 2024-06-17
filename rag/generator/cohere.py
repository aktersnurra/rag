import os
from dataclasses import asdict
from typing import Any, Generator, List

import cohere
from loguru import logger as log

from rag.rag import Message

from .abstract import AbstractGenerator
from .prompt import Prompt


class Cohere(metaclass=AbstractGenerator):
    def __init__(self) -> None:
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])

    def generate(
        self, prompt: Prompt, messages: List[Message]
    ) -> Generator[Any, Any, Any]:
        log.debug("Generating answer from cohere...")
        for event in self.client.chat_stream(
            message=prompt.to_str(),
            documents=[asdict(d) for d in prompt.documents],
            chat_history=[m.as_dict() for m in messages],
            prompt_truncation="AUTO",
        ):
            if event.event_type == "text-generation":
                yield event.text
            elif event.event_type == "citation-generation":
                yield event.citations
            elif event.event_type == "stream-end":
                yield event.finish_reason
