import json
import os
import ssl
import time
from pathlib import Path
from typing import Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

# Disable SSL verification for HuggingFace downloads
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context
_original_init = httpx.Client.__init__
def _ssl_patched_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _original_init(self, *args, **kwargs)
httpx.Client.__init__ = _ssl_patched_init

from .config import (
    PHASE_1_DIR, QDRANT_PATH, QDRANT_COLLECTION,
    EMBEDDING_MODEL, EMBEDDING_DIM,
)


def get_embedder(model_name: Optional[str] = None):
    name = model_name or EMBEDDING_MODEL
    print(f"Loading embedding model: {name} ...")
    t0 = time.time()
    model = SentenceTransformer(name, trust_remote_code=True)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def load_chunks() -> list[dict]:
    path = PHASE_1_DIR / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Run chunking first: {path} does not exist")
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def get_qdrant_client() -> QdrantClient:
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    return client


def recreate_collection(client: QdrantClient):
    try:
        client.delete_collection(QDRANT_COLLECTION)
    except Exception:
        pass

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="doc_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="element_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="domain",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="source_doc",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    print(f"Collection '{QDRANT_COLLECTION}' created with indexes")


def embed_and_index(
    chunks: list[dict],
    model: SentenceTransformer,
    client: QdrantClient,
    batch_size: int = 64,
):
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False)

        points = []
        for j, (emb, meta) in enumerate(zip(embeddings, batch_metas)):
            points.append(
                models.PointStruct(
                    id=i + j,
                    vector=emb.tolist(),
                    payload={
                        "text": batch_texts[j],
                        **meta,
                    },
                )
            )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True,
        )

        if (i // batch_size) % 5 == 0:
            print(f"  Indexed {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    elapsed = time.time() - t0
    print(f"Indexing complete: {len(texts)} chunks in {elapsed:.1f}s")


def run_indexing():
    print("=" * 60)
    print("Phase 2: Embedding & Indexing")
    print("=" * 60)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from phase 1")

    model = get_embedder()
    client = get_qdrant_client()
    recreate_collection(client)
    embed_and_index(chunks, model, client)

    count = client.count(QDRANT_COLLECTION)
    print(f"Collection '{QDRANT_COLLECTION}' has {count.count} vectors")
    return client


if __name__ == "__main__":
    run_indexing()
