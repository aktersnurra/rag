from typing import Type

from .abstract import AbstractGenerator
from .cohere import Cohere
from .ollama import Ollama

MODELS = ["ollama", "cohere"]

def get_generator(model: str) -> Type[AbstractGenerator]:
    match model:
        case "ollama":
            return Ollama()
        case "cohere":
            return Cohere()
        case _:
            exit(1)
