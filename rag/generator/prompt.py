from dataclasses import dataclass
from typing import List

from rag.retriever.vector import Document

ANSWER_INSTRUCTION = (
    "Given the context information and not prior knowledge, answer the question."
    "If the context is irrelevant to the question, answer that you do not know "
    "the answer to the question given the context and stop.\n"
)


@dataclass
class Prompt:
    query: str
    documents: List[Document]
