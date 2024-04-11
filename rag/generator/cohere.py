import os
from dataclasses import asdict
from typing import Any, Generator

import cohere
from loguru import logger as log

from .abstract import AbstractGenerator
from .prompt import ANSWER_INSTRUCTION, Prompt


class Cohere(metaclass=AbstractGenerator):
    def __init__(self) -> None:
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])

    def generate(self, prompt: Prompt) -> Generator[Any, Any, Any]:
        log.debug("Generating answer from cohere")
        query = f"{prompt.query}\n\n{ANSWER_INSTRUCTION}"
        for event in self.client.chat_stream(
            message=query,
            documents=[asdict(d) for d in prompt.documents],
            prompt_truncation="AUTO",
        ):
            if event.event_type == "text-generation":
                yield event.text
            elif event.event_type == "citation-generation":
                yield event.citations
            elif event.event_type == "stream-end":
                yield event.finish_reason
