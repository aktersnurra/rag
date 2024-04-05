from pathlib import Path
from typing import List, Optional

from langchain_core.documents.base import Document
from llm.encoder import Encoder
from llm.generator import Generator
from parser import pdf
from db.documents import Documents
from db.vectors import Vectors


class RAG:
    def __init__(self) -> None:
        self.generator = Generator()
        self.encoder = Encoder()
        self.docs = Documents()
        self.vectors = Vectors()

    def add_pdf(self, filepath: Path) -> Optional[List[Document]]:
        chunks = pdf.parser(filepath)
