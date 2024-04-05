import hashlib
import os
from typing import List

import psycopg
from langchain_core.documents.base import Document
from loguru import logger as log

TABLES = """
CREATE TABLE IF NOT EXISTS document (
        hash text PRIMARY KEY)
"""


class DocumentDB:
    def __init__(self) -> None:
        self.conn = psycopg.connect(
            f"dbname={os.environ['RAG_DB_NAME']} user={os.environ['RAG_DB_USER']}"
        )
        self.__configure()

    def close(self):
        self.conn.close()

    def __configure(self):
        log.debug("Creating documents table if it does not exist...")
        with self.conn.cursor() as cur:
            cur.execute(TABLES)
            self.conn.commit()

    def __hash(self, chunks: List[Document]) -> str:
        log.debug("Hashing document...")
        document = str.encode("".join([chunk.page_content for chunk in chunks]))
        return hashlib.sha256(document).hexdigest()

    def add(self, chunks: List[Document]) -> bool:
        with self.conn.cursor() as cur:
            hash = self.__hash(chunks)
            cur.execute(
                """
                        SELECT * FROM document 
                        WHERE
                        hash = %s
                        """,
                (hash,),
            )
            exist = cur.fetchone()
            if exist is None:
                log.debug("Inserting document hash into documents db...")
                cur.execute(
                    """
                            INSERT INTO document 
                            (hash) VALUES 
                            (%s)
                            """,
                    (hash,),
                )
            self.conn.commit()
        return exist is None
