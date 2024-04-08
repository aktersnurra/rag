from dataclasses import dataclass
from typing import List


from rag.retriever.vector import Document


@dataclass
class Prompt:
    query: str
    documents: List[Document]
