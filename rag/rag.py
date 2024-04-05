from pathlib import Path
from typing import List

from dotenv import load_dotenv
from loguru import logger as log
from qdrant_client.models import StrictFloat

from rag.db.document import DocumentDB
from rag.db.vector import VectorDB
from rag.llm.encoder import Encoder
from rag.llm.generator import Generator, Prompt
from rag.parser import pdf


class RAG:
    def __init__(self) -> None:
        load_dotenv()
        self.generator = Generator()
        self.encoder = Encoder()
        self.document_db = DocumentDB()
        self.vector_db = VectorDB()

    def add_pdf(self, filepath: Path):
        chunks = pdf.parser(filepath)
        added = self.document_db.add_document(chunks)
        if added:
            log.debug(f"Adding pdf with filepath: {filepath} to vector db")
            points = self.encoder.encode_document(chunks)
            self.vector_db.add(points)

    def __context(self, query_emb: List[StrictFloat], limit: int) -> str:
        hits = self.vector_db.search(query_emb, limit)
        log.debug(f"Got {len(hits)} hits in the vector db with limit={limit}")
        return "\n".join(h.payload["text"] for h in hits)

    def rag(self, query: str, role: str, limit: int = 5) -> str:
        query_emb = self.encoder.encode_query(query)
        context = self.__context(query_emb, limit)
        prompt = Prompt(query, context)
        return self.generator.generate(prompt, role)["response"]
