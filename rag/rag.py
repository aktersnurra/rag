from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from loguru import logger as log
from qdrant_client.models import StrictFloat


try:
    from rag.db.vector import VectorDB
    from rag.db.document import DocumentDB
    from rag.llm.encoder import Encoder
    from rag.llm.generator import Generator, Prompt
    from rag.parser.pdf import PDFParser
except ModuleNotFoundError:
    from db.vector import VectorDB
    from db.document import DocumentDB
    from llm.encoder import Encoder
    from llm.generator import Generator, Prompt
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
        self.generator = Generator()
        self.encoder = Encoder()
        self.vector_db = VectorDB()
        self.doc_db = DocumentDB()

    def add_pdf_from_path(self, path: Path):
        blob = self.pdf_parser.from_path(path)
        self.add_pdf_from_blob(blob)

    def add_pdf_from_blob(self, blob: BytesIO):
        if self.doc_db.add(blob):
            log.debug("Adding pdf to vector database...")
            chunks = self.pdf_parser.from_data(blob)
            points = self.encoder.encode_document(chunks)
            self.vector_db.add(points)
        else:
            log.debug("Document already exists!")

    def __context(self, query_emb: List[StrictFloat], limit: int) -> str:
        hits = self.vector_db.search(query_emb, limit)
        log.debug(f"Got {len(hits)} hits in the vector db with limit={limit}")
        return [h.payload["text"] for h in hits]

    def retrive(self, query: str, limit: int = 5) -> Response:
        query_emb = self.encoder.encode_query(query)
        context = self.__context(query_emb, limit)
        prompt = Prompt(query, "\n".join(context))
        answer = self.generator.generate(prompt)["response"]
        return Response(query, context, answer)
