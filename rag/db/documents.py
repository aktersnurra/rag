import hashlib
import os
from typing import List

import psycopg
from langchain_core.documents.base import Document

TABLES = """
CREATE TABLE IF NOT EXISTS document (
        hash text PRIMARY KEY)
"""


class Documents:
    def __init__(self) -> None:
        self.conn = psycopg.connect(
            f"dbname={os.environ['RAG_DB_NAME']} user={os.environ['RAG_DB_USER']}"
        )
        self.__create_content_table()

    def close(self):
        self.conn.close()

    def __create_content_table(self):
        with self.conn.cursor() as cur:
            cur.execute(TABLES)
            self.conn.commit()

    def __hash(self, chunks: List[Document]) -> str:
        document = str.encode("".join([chunk.page_content for chunk in chunks]))
        return hashlib.sha256(document).hexdigest()

    def add_document(self, chunks: List[Document]) -> bool:
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
                cur.execute(
                    """
                            INSERT INTO document 
                            (hash) VALUES 
                            (%s)
                            """,
                    (hash,),
                )
            self.conn.commit()
        return exist is not None
