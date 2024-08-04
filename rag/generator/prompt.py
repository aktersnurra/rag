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
    client: str

    def __context(self, documents: List[Document]) -> str:
        results = [
            f"Document: {i}\ntitle: {doc.title}\ntext: {doc.text}"
            for i, doc in enumerate(documents)
        ]
        return "\n".join(results)

    def to_str(self) -> str:
        if self.client == "cohere":
            return f"{self.query}\n\n{ANSWER_INSTRUCTION}"
        else:
            return (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                "Using the information contained in the context, give a comprehensive answer to the question.\n"
                "If the answer cannot be deduced from the context, do not give an answer.\n\n"
                "Context:\n"
                "---\n"
                f"{self.__context(self.documents)}\n\n"
                "---\n"
                f"Question: {self.query}<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
