import json
import re
from pathlib import Path
from typing import Any

from .config import PHASE_0_JSON, PHASE_1_DIR, CHUNK_SIZE


def load_elements() -> list[dict]:
    with open(PHASE_0_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data["elements"]


def make_text(element: dict) -> str:
    et = element.get("element_type", "")
    if et == "term_definition":
        parts = [f"Term: {element.get('term', '')}"]
        if element.get("full_form"):
            parts.append(f"Full Form: {element['full_form']}")
        if element.get("explanation"):
            parts.append(f"Explanation: {element['explanation']}")
        if element.get("example"):
            parts.append(f"Example: {element['example']}")
        if element.get("why_needed"):
            parts.append(f"Why Needed: {element['why_needed']}")
        if element.get("used_by"):
            parts.append(f"Used By: {', '.join(element['used_by'])}")
        return "\n".join(parts)

    if et == "scenario":
        parts = [
            f"Scenario {element.get('scenario_number', '')}: {element.get('title', '')}",
        ]
        if element.get("domain"):
            parts.append(f"Domain: {element['domain']}")
        if element.get("scenario_text"):
            parts.append(f"Scenario: {element['scenario_text']}")
        if element.get("what_went_wrong"):
            parts.append(f"What Went Wrong: {element['what_went_wrong']}")
        if element.get("who_was_involved"):
            parts.append(f"Who Was Involved: {element['who_was_involved']}")
        if element.get("expected_vs_actual"):
            parts.append(f"Expected vs Actual: {element['expected_vs_actual']}")
        if element.get("impact"):
            parts.append(f"Impact: {element['impact']}")
        return "\n".join(parts)

    if et == "qa_pair":
        return f"Q: {element.get('question', '')}\nA: {element.get('answer', '')}"

    if et == "section_heading":
        return element.get("text", element.get("section_title", ""))

    if et == "slide_heading":
        txt = element.get("text", "")
        proc = element.get("process_name", "")
        return f"{proc} - {txt}" if proc else txt

    if et == "narrator":
        return element.get("narrator_text", "")

    if et == "paragraph":
        txt = element.get("text", "")
        proc = element.get("process_name", "")
        slide = element.get("slide", "")
        parts = []
        if proc:
            parts.append(f"[{proc}]")
        if slide:
            parts.append(f"[{slide}]")
        parts.append(txt)
        return " ".join(parts)

    if et == "process_section":
        return element.get("text", element.get("process_name", ""))

    if et == "section_label":
        return element.get("label", "")

    return element.get("text", "")


def chunk_text(text: str, max_words: int = CHUNK_SIZE) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


def build_metadata(element: dict) -> dict:
    meta = {
        "source_doc": element.get("source_doc", ""),
        "doc_type": element.get("doc_type", ""),
        "element_type": element.get("element_type", ""),
    }
    et = element.get("element_type", "")

    if et == "term_definition":
        meta["term"] = element.get("term", "")
        meta["domain"] = "Glossary"
        meta["section"] = element.get("section_title", "")

    elif et == "scenario":
        meta["scenario_number"] = element.get("scenario_number", 0)
        meta["title"] = element.get("title", "")
        meta["domain"] = element.get("domain", "Scenario")

    elif et == "qa_pair":
        meta["question_number"] = element.get("question_number", 0)
        meta["section"] = element.get("section", "")
        meta["domain"] = "FAQ"

    elif et in ("slide_heading", "narrator", "paragraph", "section_label", "process_section"):
        meta["process_name"] = element.get("process_name", "")
        meta["slide"] = element.get("slide", "")
        if et == "slide_heading":
            meta["slide_number"] = element.get("slide_number", 0)
            meta["domain"] = "Process Flow"
        elif et == "narrator":
            meta["domain"] = "Process Flow"
        else:
            meta["domain"] = element.get("doc_type", "General")

    elif et == "section_heading":
        meta["section_title"] = element.get("section_title", "")
        meta["domain"] = "Glossary"

    return meta


def chunk_all() -> list[dict]:
    elements = load_elements()
    PHASE_1_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []

    for element in elements:
        text = make_text(element)
        if not text.strip():
            continue

        meta = build_metadata(element)
        et = element.get("element_type", "")

        if et == "term_definition":
            chunks.append({"text": text, "metadata": meta})
        elif et == "scenario":
            chunks.append({"text": text, "metadata": meta})
        elif et == "qa_pair":
            chunks.append({"text": text, "metadata": meta})
        else:
            text_chunks = chunk_text(text)
            for i, tc in enumerate(text_chunks):
                m = dict(meta)
                if len(text_chunks) > 1:
                    m["chunk_index"] = i
                    m["total_chunks"] = len(text_chunks)
                chunks.append({"text": tc, "metadata": m})

    out_path = PHASE_1_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Chunking complete: {len(chunks)} chunks written to {out_path}")
    return chunks


if __name__ == "__main__":
    chunk_all()
