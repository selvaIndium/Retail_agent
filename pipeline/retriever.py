import json
import os
import ssl
from typing import Optional

import httpx
from groq import Groq
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
    GROQ_API_KEY, GROQ_MODEL, TOP_K,
)


VALID_DOC_TYPES = {"scenarios", "cheatsheet", "faq", "process_flow", "training"}

FILTER_PROMPT = """You are a retail knowledge base classifier. Given a user question, determine which document type it is most likely asking about.

Valid document types:
- "scenarios" — Real-world problem scenarios (e.g., "What happens when...", "How to handle...", troubleshooting)
- "cheatsheet" — Glossary terms and definitions (e.g., "What is...", "Define...", "Explain a term")
- "faq" — Frequently asked questions about retail concepts
- "process_flow" — Step-by-step processes (e.g., "How does... work", "Steps in...", "Walk me through...")
- "training" — General retail training content

If the question is general or doesn't clearly target one type, return null for filter_doc_type.

Respond with valid JSON only:
{{"filter_doc_type": "<doc_type or null>", "reasoning": "<brief explanation>"}}

Question: {question}"""


class SelfQueryRetriever:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder: Optional[SentenceTransformer] = None,
        groq_client: Optional[Groq] = None,
        model: Optional[str] = None,
    ):
        self.client = client or QdrantClient(path=str(QDRANT_PATH))
        self.embedder = embedder or SentenceTransformer(
            EMBEDDING_MODEL, trust_remote_code=True
        )
        self.llm_client = groq_client or (Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None)
        self.llm_model = model or GROQ_MODEL

    def extract_filters(self, question: str) -> dict:
        if not self.llm_client:
            return {}
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": FILTER_PROMPT.format(question=question)}],
                temperature=0,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            doc_type = result.get("filter_doc_type")
            reasoning = result.get("reasoning", "")
            if doc_type and doc_type in VALID_DOC_TYPES:
                print(f"[Filter] Question: {question} -> doc_type={doc_type} ({reasoning})")
                return {"doc_type": doc_type}
            else:
                print(f"[Filter] Question: {question} -> no filter ({reasoning})")
        except Exception as e:
            print(f"[Filter] Error classifying question '{question}': {e}")
        return {}

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
