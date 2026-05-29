import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import json
import ssl
import time
from pathlib import Path

import httpx
import streamlit as st

# SSL patch for HuggingFace downloads
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context
_original_init = httpx.Client.__init__

def _ssl_patched_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _original_init(self, *args, **kwargs)
httpx.Client.__init__ = _ssl_patched_init

from pipeline.chunking import chunk_all
from pipeline.indexing import (
    load_chunks, get_embedder, get_qdrant_client,
    recreate_collection, embed_and_index,
)
from pipeline.retriever import SelfQueryRetriever
from pipeline.generator import AnswerGenerator
from pipeline.config import GROQ_API_KEY

st.set_page_config(
    page_title="Retail Knowledge Bot",
    page_icon="🛒",
    layout="wide",
)

for key in ["retriever", "generator", "indexed", "messages", "api_key_set"]:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ("indexed", "api_key_set") else False
if st.session_state.messages is None:
    st.session_state.messages = []

@st.cache_resource
def get_qdrant():
    from pipeline.config import QDRANT_PATH
    _lock_file = QDRANT_PATH / ".lock"
    if _lock_file.exists():
        try:
            _lock_file.unlink()
        except Exception:
            pass
    return get_qdrant_client()

@st.cache_resource
def get_embeddings_model():
    return get_embedder()

def try_load_index():
    from pipeline.config import QDRANT_PATH, QDRANT_COLLECTION
    if not (QDRANT_PATH / "collection" / QDRANT_COLLECTION).exists():
        st.info("No existing index found. Click 'Build Index' to create one.")
        return False
    try:
        client = get_qdrant()
        count = client.count(QDRANT_COLLECTION)
        if count.count == 0:
            return False
        model = get_embeddings_model()
        retriever = SelfQueryRetriever(client=client, embedder=model)
        st.session_state.retriever = retriever
        st.session_state.indexed = True
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            st.session_state.generator = AnswerGenerator(api_key=api_key)
            st.session_state.api_key_set = True
        return True
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return False

if not st.session_state.indexed:
    try_load_index()

with st.sidebar:
    st.title("🛒 Retail KB")
    st.markdown("---")

    if st.button("Build Index (Phase 1+2)", use_container_width=True):
        with st.spinner("Chunking documents..."):
            chunk_all()
        with st.spinner("Loading embedding model..."):
            chunks = load_chunks()
            model = get_embeddings_model()
        with st.spinner("Indexing into Qdrant..."):
            client = get_qdrant()
            recreate_collection(client)
            embed_and_index(chunks, model, client)
        st.session_state.indexed = True
        retriever = SelfQueryRetriever(client=client, embedder=model)
        st.session_state.retriever = retriever
        generator = AnswerGenerator(api_key=os.environ.get("GROQ_API_KEY", ""))
        st.session_state.generator = generator
        count = client.count("retail_chunks")
        st.success(f"Indexed {count.count} chunks ready!")

    st.metric("Index", "Ready" if st.session_state.indexed else "Not built")
    st.metric("API Key", "Set" if os.environ.get("GROQ_API_KEY") else "Missing (.env)")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Retail Knowledge Assistant")
st.caption("Ask questions about retail operations, scenarios, terms, and processes")

if not st.session_state.api_key_set:
    st.info("Set your Groq API key in the .env file to enable AI answers.")
if not st.session_state.indexed:
    st.info("Click 'Build Index' in the sidebar to index the retail knowledge base.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    parts = []
                    if s.get("title"):
                        parts.append(f"Scenario: {s['title']}")
                    if s.get("term"):
                        parts.append(f"Term: {s['term']}")
                    if s.get("domain"):
                        parts.append(f"Domain: {s['domain']}")
                    label = f"**{s['source_doc']}**"
                    if parts:
                        label += " - " + " | ".join(parts)
                    st.markdown(f"- {label} (score: {s['relevance_score']})")

def answer_question(prompt):
    if not st.session_state.api_key_set:
        with st.chat_message("assistant"):
            st.error("Please set your Groq API key in the .env file first.")
        st.session_state.messages.append({"role": "assistant", "content": "API key missing."})
        return

    if not st.session_state.indexed:
        with st.chat_message("assistant"):
            st.error("Please build the index first using the sidebar button.")
        st.session_state.messages.append({"role": "assistant", "content": "Index not built."})
        return

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            t0 = time.time()
            try:
                chunks = st.session_state.retriever.retrieve(prompt)
                answer, sources = st.session_state.generator.generate_with_sources(prompt, chunks)
                elapsed = time.time() - t0
                st.markdown(answer)
                st.caption(f"Retrieved in {elapsed:.1f}s from {len(chunks)} chunks")
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            parts = []
                            if s.get("title"):
                                parts.append(f"Scenario: {s['title']}")
                            if s.get("term"):
                                parts.append(f"Term: {s['term']}")
                            if s.get("domain"):
                                parts.append(f"Domain: {s['domain']}")
                            label = f"**{s['source_doc']}**"
                            if parts:
                                label += " - " + " | ".join(parts)
                            st.markdown(f"- {label} (score: {s['relevance_score']})")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                })

if st.session_state.get("pending_question"):
    prompt = st.session_state.pop("pending_question")
    with st.chat_message("user"):
        st.markdown(prompt)
    answer_question(prompt)

if prompt := st.chat_input("Ask about retail..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    answer_question(prompt)
