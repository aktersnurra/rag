from abc import abstractmethod
from typing import Any, Generator, List

from rag.message import Message
from rag.retriever.vector import Document


class AbstractGenerator(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @abstractmethod
    def generate(self, messages: List[Message], documents: List[Document]) -> Generator[Any, Any, Any]:
        pass
