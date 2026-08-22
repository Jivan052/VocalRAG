"""
Offline chunking. This is prep work that runs once when you build the index,
never on the query path, so we favor clarity over speed here.

Fixed+overlap is the simplest strategy and is what's wired up by default.
Semantic-split and parent-child hierarchy are stubbed below so you can swap
strategies later without touching the indexing script.
"""
from typing import List, Dict
from app.config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


def fixed_overlap_chunks(doc_id: str, title: str, text: str,
                          size: int = CHUNK_SIZE_CHARS,
                          overlap: int = CHUNK_OVERLAP_CHARS) -> List[Dict]:
    """Split text into overlapping fixed-size windows, snapped to sentence
    boundaries where possible so we don't cut a claim in half mid-sentence."""
    chunks = []
    start = 0
    n = len(text)
    idx = 0
    while start < n:
        end = min(start + size, n)
        # try to extend to the next sentence boundary within a small lookahead
        if end < n:
            lookahead = text[end:end + 60]
            period = lookahead.find(". ")
            if period != -1:
                end += period + 1
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": f"{doc_id}::{idx}",
                "doc_id": doc_id,
                "title": title,
                "text": chunk_text,
            })
            idx += 1
        if end >= n:
            break
        start = end - overlap  # step forward, keeping overlap
    return chunks


def semantic_split_chunks(doc_id: str, title: str, text: str) -> List[Dict]:
    """Placeholder for embedding-similarity-based splitting (break where
    consecutive-sentence similarity drops). Not wired up by default —
    fixed+overlap is enough for a corpus this size."""
    raise NotImplementedError("Swap in when corpus size / topic drift justifies it")


def parent_child_chunks(doc_id: str, title: str, text: str) -> List[Dict]:
    """Placeholder for parent-child hierarchy: embed small child chunks for
    precision, but return the larger parent chunk as LLM context for recall."""
    raise NotImplementedError("Swap in if generation quality suffers from tiny chunks")


def build_chunks_from_corpus(corpus: List[Dict]) -> List[Dict]:
    all_chunks = []
    for doc in corpus:
        all_chunks.extend(
            fixed_overlap_chunks(doc["id"], doc.get("title", ""), doc["text"])
        )
    return all_chunks
