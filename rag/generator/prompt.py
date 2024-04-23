from dataclasses import dataclass
from typing import List

from rag.retriever.vector import Document

ANSWER_INSTRUCTION = (
    "Given the context information and not prior knowledge, answer the question."
    "If the context is irrelevant to the question or empty, then do not attempt to answer "
    "the question, just reply that you do not know based on the context provided.\n"
)


@dataclass
class Prompt:
    query: str
    documents: List[Document]
