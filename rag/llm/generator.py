import os
from dataclasses import dataclass

import ollama


@dataclass
class Prompt:
    question: str
    context: str

    # def context(self) -> str:
    #     return "\n".join(point.payload["text"] for point in self.points)


class Generator:
    def __init__(self) -> None:
        self.model = os.environ["GENERATOR_MODEL"]

    def __metaprompt(self, role: str, prompt: Prompt) -> str:
        metaprompt = (
            f"You are a {role}.\n"
            "Answer the following question using the provided context.\n"
            "If you can't find the answer, do not pretend you know it,"
            'but answer "I don\'t know".'
            f"Question: {prompt.question.strip()}\n\n"
            "Context:\n"
            f"{prompt.context.strip()}\n\n"
            "Answer:\n"
        )
        return metaprompt

    def generate(self, role: str, prompt: Prompt) -> str:
        metaprompt = self.__metaprompt(role, prompt)
        return ollama.generate(model=self.model, prompt=metaprompt)
