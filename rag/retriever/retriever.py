from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from loguru import logger as log

from .document import DocumentDB
from .encoder import Encoder, Query
from .parser.pdf import PDFParser
from .vector import Document, VectorDB


@dataclass
class FilePath:
    path: Path


@dataclass
class Blob:
    blob: BytesIO
    source: Optional[str] = None


FileType = FilePath | Blob


class Retriever:
    def __init__(self) -> None:
        self.pdf_parser = PDFParser()
        self.encoder = Encoder()
        self.doc_db = DocumentDB()
        self.vec_db = VectorDB()

    def __index_pdf_from_path(self, path: Path):
        log.debug(f"Indexing pdf from {path}")
        blob = self.pdf_parser.from_path(path)
        self.__index_pdf_from_blob(blob, None)

    def __index_pdf_from_blob(self, blob: BytesIO, source: Optional[str]):
        if self.doc_db.create(blob):
            log.debug("Indexing pdf to vector database...")
            document = self.pdf_parser.from_data(blob)
            chunks = self.pdf_parser.chunk(document, source)
            points = self.encoder.encode(chunks)
            self.vec_db.index(points)
        else:
            log.debug("Document already exists!")

    def index(self, filetype: FileType):
        match filetype:
            case FilePath(path):
                self.__index_pdf_from_path(path)
            case Blob(blob, source):
                self.__index_pdf_from_blob(blob, source)

    def search(self, query: Query) -> List[Document]:
        log.debug(f"Finding documents matching query: {query.query}")
        query_emb = self.encoder.encode(query)
        return self.vec_db.search(query_emb)
