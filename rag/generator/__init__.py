from typing import Type

from .abstract import AbstractGenerator
from .ollama import Ollama
from .cohere import Cohere

MODELS = ["ollama", "cohere"]

def get_generator(model: str) -> Type[AbstractGenerator]:
    match model:
        case "ollama":
            return Ollama()
        case "cohere":
            return Cohere()
        case _:
            exit(1)
