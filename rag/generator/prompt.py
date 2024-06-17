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
    generator_model: str

    def __context(self, documents: List[Document]) -> str:
        results = [
            f"Document: {i}\ntitle: {doc.title}\ntext: {doc.text}"
            for i, doc in enumerate(documents)
        ]
        return "\n".join(results)

    def to_str(self) -> str:
        if self.generator_model == "cohere":
            return f"{self.query}\n\n{ANSWER_INSTRUCTION}"
        else:
            return (
                "Context information is below.\n"
                "---------------------\n"
                f"{self.__context(self.documents)}\n\n"
                "---------------------\n"
                f"{ANSWER_INSTRUCTION}"
                "Do not attempt to answer the query without relevant context and do not use"
                " prior knowledge or training data!\n"
                f"Query: {self.query.strip()}\n\n"
                "Answer:"
            )
