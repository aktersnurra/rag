import os
from pathlib import Path
from typing import Iterator, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.document_loaders.parsers.pdf import (
    PyPDFParser,
)
from rag.db.document import DocumentDB


class PDF:
    def __init__(self) -> None:
        self.db = DocumentDB()
        self.parser = PyPDFParser(password=None, extract_images=False)

    def from_data(self, blob) -> Optional[Iterator[Document]]:
        if self.db.add(blob):
            yield from self.parser.parse(blob)
        yield None

    def from_path(self, file_path: Path) -> Optional[Iterator[Document]]:
        blob = Blob.from_path(file_path)
        from_data(blob)

    def chunk(self, content: Iterator[Document]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.environ["CHUNK_SIZE"]),
            chunk_overlap=int(os.environ["CHUNK_OVERLAP"]),
        )
        chunks = splitter.split_documents(content)
        return chunks
