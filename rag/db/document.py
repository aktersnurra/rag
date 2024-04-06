import hashlib
import os

import psycopg
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

    def __hash(self, blob: bytes) -> str:
        log.debug("Hashing document...")
        return hashlib.sha256(blob).hexdigest()

    def add(self, blob: bytes) -> bool:
        with self.conn.cursor() as cur:
            hash = self.__hash(blob)
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
