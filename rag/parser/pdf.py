import os
from pathlib import Path
from typing import Iterator

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders.parsers.pdf import (
    PyPDFParser,
)
from langchain_community.document_loaders.blob_loaders import Blob


class PDFParser:
    def __init__(self) -> None:
        self.parser = PyPDFParser(password=None, extract_images=False)

    def from_data(self, blob: Blob) -> Iterator[Document]:
        yield from self.parser.parse(blob)

    def from_path(self, path: Path) -> Iterator[Document]:
        return Blob.from_path(path)

    def chunk(self, content: Iterator[Document]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.environ["CHUNK_SIZE"]),
            chunk_overlap=int(os.environ["CHUNK_OVERLAP"]),
        )
        chunks = splitter.split_documents(content)
        return chunks
