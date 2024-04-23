from dataclasses import dataclass
from typing import List

from rag.retriever.vector import Document

ANSWER_INSTRUCTION = (
    "Given the context information and not prior knowledge, answer the query."
    "If the context is irrelevant to the query or empty, then do not attempt to answer "
    "the query, just reply that you do not know based on the context provided.\n"
)


@dataclass
class Prompt:
    query: str
    documents: List[Document]
