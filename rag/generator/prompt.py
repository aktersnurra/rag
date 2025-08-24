from dataclasses import dataclass
from typing import List

from rag.retriever.vector import Document

ANSWER_INSTRUCTION = (
    "Using the information contained in the context, give a comprehensive answer to the question.\n"
    "If the answer cannot be deduced from the context, do not give an answer.\n\n"
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
                f"{ANSWER_INSTRUCTION}"
                "Context:\n"
                "---\n"
                f"{self.__context(self.documents)}\n\n"
                "---\n"
                f"Question: {self.query}<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
