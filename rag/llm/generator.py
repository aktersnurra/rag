import os
from dataclasses import dataclass

import ollama
from loguru import logger as log


@dataclass
class Prompt:
    query: str
    context: str


class Generator:
    def __init__(self) -> None:
        self.model = os.environ["GENERATOR_MODEL"]

    def __metaprompt(self, role: str, prompt: Prompt) -> str:
        metaprompt = (
            f"You are a {role}.\n"
            "Answer the following question using the provided context.\n"
            "If you can't find the answer, do not pretend you know it,"
            'but answer "I don\'t know".\n\n'
            f"Question: {prompt.query.strip()}\n\n"
            "Context:\n"
            f"{prompt.context.strip()}\n\n"
            "Answer:\n"
        )
        return metaprompt

    def generate(self, prompt: Prompt, role: str) -> str:
        log.debug("Generating answer...")
        metaprompt = self.__metaprompt(role, prompt)
        print(f"metaprompt = \n{metaprompt}")
        return ollama.generate(model=self.model, prompt=metaprompt)
