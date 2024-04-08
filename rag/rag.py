from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from loguru import logger as log



try:
    from rag.db.vector import VectorDB, Document
    from rag.db.document import DocumentDB
    from rag.llm.encoder import Encoder
    from rag.llm.ollama_generator import OllamaGenerator, Prompt
    from rag.llm.cohere_generator import CohereGenerator
    from rag.parser.pdf import PDFParser
except ModuleNotFoundError:
    from db.vector import VectorDB, Document
    from db.document import DocumentDB
    from llm.encoder import Encoder
    from llm.ollama_generator import OllamaGenerator, Prompt
    from llm.cohere_generator import CohereGenerator
    from parser.pdf import PDFParser


@dataclass
class Response:
    query: str
    context: List[str]
    answer: str


class RAG:
    def __init__(self) -> None:
        # FIXME: load this somewhere else?
        load_dotenv()
        self.pdf_parser = PDFParser()
        self.generator = CohereGenerator()
        self.encoder = Encoder()
        self.vector_db = VectorDB()
        self.doc_db = DocumentDB()

    def add_pdf_from_path(self, path: Path):
        blob = self.pdf_parser.from_path(path)
        self.add_pdf_from_blob(blob)

    def add_pdf_from_blob(self, blob: BytesIO, source: str):
        if self.doc_db.add(blob):
            log.debug("Adding pdf to vector database...")
            document = self.pdf_parser.from_data(blob)
            chunks = self.pdf_parser.chunk(document, source)
            points = self.encoder.encode_document(chunks)
            self.vector_db.add(points)
        else:
            log.debug("Document already exists!")

    def search(self, query: str, limit: int = 5) -> List[Document]:
        query_emb = self.encoder.encode_query(query)
        return self.vector_db.search(query_emb, limit)

    def retrieve(self, prompt: Prompt):
        yield from self.generator.generate(prompt)
