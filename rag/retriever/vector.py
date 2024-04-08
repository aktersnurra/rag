import os
from dataclasses import dataclass
from typing import Dict, List

from loguru import logger as log
from qdrant_client import QdrantClient
from qdrant_client.http.models import StrictFloat
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass
class Point:
    id: str
    vector: List[StrictFloat]
    payload: Dict[str, str]


@dataclass
class Document:
    title: str
    text: str


class VectorDB:
    def __init__(self, score_threshold: float = 0.6):
        self.dim = int(os.environ["EMBEDDING_DIM"])
        self.collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        self.client = QdrantClient(url=os.environ["QDRANT_URL"])
        self.score_threshold = score_threshold
        self.__configure()

    def __configure(self):
        collections = list(
            map(lambda col: col.name, self.client.get_collections().collections)
        )
        if self.collection_name not in collections:
            log.debug(f"Configuring collection {self.collection_name}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
        else:
            log.debug(f"Collection {self.collection_name} already exists...")

    def add(self, points: List[Point]):
        log.debug(f"Inserting {len(points)} vectors into the vector db...")
        self.client.upload_points(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
            parallel=4,
            max_retries=3,
        )

    def search(self, query: List[float], limit: int = 5) -> List[Document]:
        log.debug("Searching for vectors...")
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query,
            limit=limit,
            score_threshold=self.score_threshold,
        )
        log.debug(f"Got {len(hits)} hits in the vector db with limit={limit}")
        return list(
            map(
                lambda h: Document(
                    title=h.payload.get("source", ""), text=h.payload["text"]
                ),
                hits,
            )
        )
