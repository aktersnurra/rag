import os
from dataclasses import dataclass
from typing import Dict, List
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import StrictFloat
from qdrant_client.models import Distance, VectorParams, PointStruct


@dataclass
class Point:
    id: str
    vector: List[StrictFloat]
    payload: Dict[str, str]


class Embeddings:
    def __init__(self):
        self.dim = int(os.environ["EMBEDDING_DIM"])
        self.collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        self.client = QdrantClient(url=os.environ["QDRANT_URL"])
        self.client.delete_collection(
            collection_name=self.collection_name,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def add(self, points: List[Point]):
        print(len(points))
        self.client.upload_points(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
            parallel=4,
            max_retries=3,
        )

    def search(self, query: List[float], limit: int = 4):
        hits = self.client.search(
            collection_name=self.collection_name, query_vector=query, limit=limit
        )
        return hits
