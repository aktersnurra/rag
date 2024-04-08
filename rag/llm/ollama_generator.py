import os
from dataclasses import dataclass
from typing import Any, Generator, List

import ollama
from loguru import logger as log

try:
    from rag.db.vector import Document
except ModuleNotFoundError:
    from db.vector import Document


@dataclass
class Prompt:
    query: str
    documents: List[Document]


SYSTEM_PROMPT = (
    "# System Preamble"
    "## Basic Rules"
    "When you answer the user's requests, you cite your sources in your answers, according to those instructions."
    "Answer the following question using the provided context.\n"
    "## Style Guide"
    "Unless the user asks for a different style of answer, you should answer "
    "in full sentences, using proper grammar and spelling."
)


class OllamaGenerator:
    def __init__(self) -> None:
        self.model = os.environ["GENERATOR_MODEL"]

    def __context(self, documents: List[Document]) -> str:
        results = [
            f"Document: {i}\ntitle: {doc.title}\n{doc.text}"
            for i, doc in enumerate(documents)
        ]
        return "\n".join(results)

    def __metaprompt(self, prompt: Prompt) -> str:
        # Include sources
        metaprompt = (
            f'Question: "{prompt.query.strip()}"\n\n'
            "Context:\n"
            "<result>\n"
            f"{self.__context(prompt.documents)}\n\n"
            "</result>\n"
            "Carefully perform the following instructions, in order, starting each "
            "with a new line.\n"
            "Firstly, Decide which of the retrieved documents are relevant to the "
            "user's last input by writing 'Relevant Documents:' followed by "
            "comma-separated list of document numbers.\n If none are relevant, you "
            "should instead write 'None'.\n"
            "Secondly, Decide which of the retrieved documents contain facts that "
            "should be cited in a good answer to the user's last input by writing "
            "'Cited Documents:' followed a comma-separated list of document numbers. "
            "If you dont want to cite any of them, you should instead write 'None'.\n"
            "Thirdly, Write 'Answer:' followed by a response to the user's last input "
            "in high quality natural english. Use the retrieved documents to help you. "
            "Do not insert any citations or grounding markup.\n"
            "Finally, Write 'Grounded answer:' followed by a response to the user's "
            "last input in high quality natural english. Use the symbols <co: doc> and "
            "</co: doc> to indicate when a fact comes from a document in the search "
            "result, e.g <co: 0>my fact</co: 0> for a fact from document 0."
        )
        return metaprompt

    def generate(self, prompt: Prompt) -> Generator[Any, Any, Any]:
        log.debug("Generating answer...")
        metaprompt = self.__metaprompt(prompt)
        for chunk in ollama.generate(
            model=self.model, prompt=metaprompt, system=SYSTEM_PROMPT, stream=True
            ):
            yield chunk
