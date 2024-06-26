import os
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
    PyPDFParser,
)
from langchain_core.documents import Document


class PDFParser:
    def __init__(self) -> None:
        self.parser = PyPDFParser(password=None, extract_images=False)

    def from_data(self, blob: Blob) -> List[Document]:
        return self.parser.parse(blob)

    def from_path(self, path: Path) -> Blob:
        return Blob.from_path(path)

    def chunk(
        self, document: List[Document], source: Optional[str] = None
    ) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.environ["CHUNK_SIZE"]),
            chunk_overlap=int(os.environ["CHUNK_OVERLAP"]),
        )
        chunks = splitter.split_documents(document)
        if source is not None:
            for c in chunks:
                c.metadata["source"] = source
        return chunks
