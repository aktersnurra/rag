from dataclasses import dataclass
from typing import List

from rag.retriever.vector import Document

ANSWER_INSTRUCTION = (
    "Do not attempt to answer the query without relevant context, and do not use "
    "prior knowledge or training data!\n"
    "If the context does not contain the answer or is empty, only reply that you "
    "cannot answer the query given the context."
)


@dataclass
class Prompt:
    query: str
    documents: List[Document]
