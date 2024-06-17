from abc import abstractmethod
from typing import List

from rag.memory import Message
from rag.retriever.vector import Document


class AbstractReranker(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @abstractmethod
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        pass

    @abstractmethod
    def rerank_messages(self, query: str, messages: List[Message]) -> List[Message]:
        pass
