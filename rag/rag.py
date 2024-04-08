from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Type

from dotenv import load_dotenv
from loguru import logger as log


try:
    from rag.retriever.vector import Document
    from rag.generator.abstract import AbstractGenerator
    from rag.retriever.retriever import Retriever
    from rag.generator.prompt import Prompt
except ModuleNotFoundError:
    from retriever.vector import Document
    from generator.abstract import AbstractGenerator
    from retriever.retriever import Retriever
    from generator.prompt import Prompt


@dataclass
class Response:
    query: str
    context: List[str]
    answer: str


class RAG:
    def __init__(self) -> None:
        # FIXME: load this somewhere else?
        load_dotenv()
        self.retriever = Retriever()

    def add_pdf(
        self,
        path: Optional[Path],
        blob: Optional[BytesIO],
        source: Optional[str] = None,
    ):
        if path:
            self.retriever.add_pdf_from_path(path)
        elif blob:
            self.retriever.add_pdf_from_blob(blob, source)
        else:
            log.error("Both path and blob was None, no pdf added!")

    def retrieve(self, query: str, limit: int = 5) -> List[Document]:
        return self.retriever.retrieve(query, limit)

    def generate(self, generator: Type[AbstractGenerator], prompt: Prompt):
        yield from generator.generate(prompt)
