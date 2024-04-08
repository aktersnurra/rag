from pathlib import Path
from typing import Optional, List
from loguru import logger as log

from io import BytesIO
from .document import DocumentDB
from .encoder import Encoder
from .parser.pdf import PDFParser
from .vector import VectorDB, Document


class Retriever:
    def __init__(self) -> None:
        self.pdf_parser = PDFParser()
        self.encoder = Encoder()
        self.doc_db = DocumentDB()
        self.vec_db = VectorDB()

    def __add_pdf_from_path(self, path: Path):
        log.debug(f"Adding pdf from {path}")
        blob = self.pdf_parser.from_path(path)
        self.__add_pdf_from_blob(blob)

    def __add_pdf_from_blob(self, blob: BytesIO, source: Optional[str] = None):
        if self.doc_db.add(blob):
            log.debug("Adding pdf to vector database...")
            document = self.pdf_parser.from_data(blob)
            chunks = self.pdf_parser.chunk(document, source)
            points = self.encoder.encode_document(chunks)
            self.vec_db.add(points)
        else:
            log.debug("Document already exists!")

    def add_pdf(
        self,
        path: Optional[Path] = None,
        blob: Optional[BytesIO] = None,
        source: Optional[str] = None,
    ):
        if path:
            self.__add_pdf_from_path(path)
        elif blob and source:
            self.__add_pdf_from_blob(blob, source)
        else:
            log.error("Invalid input!")

    def retrieve(self, query: str, limit: int = 5) -> List[Document]:
        log.debug(f"Finding documents matching query: {query}")
        query_emb = self.encoder.encode_query(query)
        return self.vec_db.search(query_emb, limit)
