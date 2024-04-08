from dataclasses import dataclass
from typing import List


try:
    from rag.retriever.vector import Document
except ModuleNotFoundError:
    from retriever.vector import Document


@dataclass
class Prompt:
    query: str
    documents: List[Document]
