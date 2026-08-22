"""
Thin wrapper around a small local embedding model.

Loaded once as a module-level singleton so the runtime query path never
pays model-load cost — only inference cost (a few ms on CPU for MiniLM).
"""
import numpy as np
from functools import lru_cache
from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns L2-normalized embeddings, shape (len(texts), dim), float32.
    Normalizing here means dense search is a plain dot product (cosine sim)."""
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vecs = vecs.astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vecs / norms


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
