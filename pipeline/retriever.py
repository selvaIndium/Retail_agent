import os
import ssl
from typing import Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context
_original_init = httpx.Client.__init__
def _ssl_patched_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _original_init(self, *args, **kwargs)
httpx.Client.__init__ = _ssl_patched_init

from .config import (
    QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_MODEL,
    TOP_K,
)


class SelfQueryRetriever:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder: Optional[SentenceTransformer] = None,
    ):
        self.client = client or QdrantClient(path=str(QDRANT_PATH))
        self.embedder = embedder or SentenceTransformer(
            EMBEDDING_MODEL, trust_remote_code=True
        )

    def extract_filters(self, question: str) -> dict:
        q_lower = question.lower()
        filters = {}

        doc_type_keywords = {
            "scenario": "scenarios",
            "scenarios": "scenarios",
            "glossary": "cheatsheet",
            "cheatsheet": "cheatsheet",
            "term": "cheatsheet",
            "definition": "cheatsheet",
            "faq": "faq",
            "process": "process_flow",
            "process flow": "process_flow",
            "training": "training",
            "slide": "process_flow",
        }
        for keyword, dt in doc_type_keywords.items():
            if keyword in q_lower:
                filters["doc_type"] = dt
                break

        return filters

    def _search(self, query_vector: list, query_filter, k: int) -> list[dict]:
        result = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
        )
        results = []
        for point in result.points:
            results.append({
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {
                    k: v for k, v in point.payload.items() if k != "text"
                },
            })
        return results

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        filters = self.extract_filters(question)
        query_embedding = self.embedder.encode(question).tolist()

        filter_conditions = []
        for key, value in filters.items():
            filter_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(
                must=filter_conditions,
            )

        return self._search(query_embedding, query_filter, k)

    def retrieve_with_filters(
        self, question: str, override_filters: dict, k: int = TOP_K
    ) -> list[dict]:
        query_embedding = self.embedder.encode(question).tolist()

        filter_conditions = []
        for key, value in override_filters.items():
            if value:
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        return self._search(query_embedding, query_filter, k)
