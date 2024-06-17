import os
from typing import Any, Generator, List

import ollama
from loguru import logger as log

from rag.rag import Message

from .abstract import AbstractGenerator
from .prompt import Prompt


class Ollama(metaclass=AbstractGenerator):
    def __init__(self) -> None:
        self.model = os.environ["GENERATOR_MODEL"]
        log.debug(f"Using {self.model} for generator...")

    def generate(
        self, prompt: Prompt, messages: List[Message]
    ) -> Generator[Any, Any, Any]:
        log.debug("Generating answer with ollama...")
        messages = messages.append(
            Message(role="user", content=prompt.to_str(), client="ollama")
        )
        messages = [m.as_dict() for m in messages]
        for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
            yield chunk["response"]
