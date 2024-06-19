from typing import Type

from rag.generator.abstract import AbstractGenerator
from rag.generator.cohere import Cohere
from rag.generator.ollama import Ollama

MODELS = ["local", "cohere"]

def get_generator(model: str) -> Type[AbstractGenerator]:
    match model:
        case "local":
            return Ollama()
        case "cohere":
            return Cohere()
        case _:
            exit(1)
