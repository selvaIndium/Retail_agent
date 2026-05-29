import json
import time
from pathlib import Path

from .chunking import chunk_all
from .indexing import load_chunks, get_embedder, get_qdrant_client, recreate_collection, embed_and_index
from .retriever import SelfQueryRetriever
from .generator import AnswerGenerator
from .config import (
    PHASE_1_DIR, OUTPUT_DIR, GROQ_API_KEY,
)


def run_phase_1():
    print("\n" + "=" * 60)
    print("Phase 1: Chunking")
    print("=" * 60)
    return chunk_all()


def run_phase_2():
    print("\n" + "=" * 60)
    print("Phase 2: Embedding & Indexing")
    print("=" * 60)
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    model = get_embedder()
    client = get_qdrant_client()
    recreate_collection(client)
    embed_and_index(chunks, model, client)

    count = client.count("retail_chunks")
    print(f"Collection has {count.count} vectors")
    return client


def run_test(retriever: SelfQueryRetriever, generator: AnswerGenerator):
    print("\n" + "=" * 60)
    print("Phase 5: Testing")
    print("=" * 60)

    from .test_set import get_test_set
    questions = get_test_set()

    results = []
    for q in questions:
        t0 = time.time()
        chunks = retriever.retrieve(q, k=5)
        answer, sources = generator.generate_with_sources(q, chunks)
        elapsed = time.time() - t0

        results.append({
            "question": q,
            "latency": round(elapsed, 2),
            "num_chunks": len(chunks),
            "avg_score": round(sum(c["score"] for c in chunks) / len(chunks), 4) if chunks else 0,
            "sources": [s["source_doc"] for s in sources],
        })
        print(f"  [{elapsed:.1f}s] {q[:60]}...")

    report_path = OUTPUT_DIR / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_questions": len(results),
            "avg_latency": round(sum(r["latency"] for r in results) / len(results), 2),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nTest report saved to {report_path}")
    avg_lat = sum(r["latency"] for r in results) / len(results)
    print(f"Average latency: {avg_lat:.2f}s")
    return results


def build_pipeline():
    """Run phases 1-4 sequentially."""
    run_phase_1()
    run_phase_2()

    print("\n" + "=" * 60)
    print("Phase 3 & 4: Retriever + Generator Ready")
    print("=" * 60)

    retriever = SelfQueryRetriever()
    generator = AnswerGenerator(api_key=GROQ_API_KEY)

    if not GROQ_API_KEY:
        print("\nWARNING: GROQ_API_KEY not set!")
        print("Set it with: $env:GROQ_API_KEY='your-key'")
        print("Get a free key at: https://console.groq.com/keys\n")

    return retriever, generator


if __name__ == "__main__":
    build_pipeline()
