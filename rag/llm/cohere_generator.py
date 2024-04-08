import os
from typing import Any, Generator
import cohere

from dataclasses import asdict
try:
    from rag.llm.ollama_generator import Prompt
except ModuleNotFoundError:
    from llm.ollama_generator import Prompt
from loguru import logger as log


class CohereGenerator:
    def __init__(self) -> None:
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])

    def generate(self, prompt: Prompt) -> Generator[Any, Any, Any]:
        log.debug("Generating answer from cohere")
        for event in self.client.chat_stream(
            message=prompt.query,
            documents=[asdict(d) for d in prompt.documents],
            prompt_truncation="AUTO",
        ):
            if event.event_type == "text-generation":
                yield event.text
            elif event.event_type == "citation-generation":
                yield event.citations
            elif event.event_type == "stream-end":
                yield event.finish_reason
