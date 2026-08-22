"""
Runtime retrieval layer. Everything here runs on the hot path, so:
  - indexes are loaded once at process startup (module-level singletons)
  - no disk I/O per query
  - the cross-encoder is loaded lazily and reused across calls
"""
import json
import pickle
import time
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import faiss

from app.config import (
    FAISS_INDEX_PATH, CHUNK_STORE_PATH, BM25_STORE_PATH,
    DENSE_TOP_K, SPARSE_TOP_K, RRF_K, FUSED_TOP_K, RERANK_TOP_K,
)
from app.embedding import embed_query


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


class RetrievalIndex:
    """Loads all three artifacts once and serves queries against them."""

    def __init__(self):
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                "Index not built yet. Run `python build_index.py` first."
            )
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.chunks = json.loads(CHUNK_STORE_PATH.read_text())
        with open(BM25_STORE_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        self._cross_encoder = None  # lazy-loaded on first rerank call

    # ---- dense leg ----
    def _dense_search(self, query_vec: np.ndarray, k: int):
        query_vec = query_vec.reshape(1, -1)
        scores, ids = self.faiss_index.search(query_vec, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    # ---- sparse leg ----
    def _sparse_search(self, query: str, k: int):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    # ---- reciprocal rank fusion ----
    @staticmethod
    def _rrf_fuse(dense_ranked, sparse_ranked, k=RRF_K):
        fused = {}
        for rank, (idx, _) in enumerate(dense_ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(sparse_ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # ---- cross-encoder rerank ----
    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder

    def _rerank(self, query: str, candidate_idxs: list[int], top_k: int):
        ce = self._get_cross_encoder()
        pairs = [(query, self.chunks[i]["text"]) for i in candidate_idxs]
        scores = ce.predict(pairs)
        order = np.argsort(scores)[::-1][:top_k]
        return [(candidate_idxs[o], float(scores[o])) for o in order]

    def retrieve(self, query: str, timings: dict | None = None) -> list[RetrievedChunk]:
        t0 = time.perf_counter()
        qvec = embed_query(query)
        if timings is not None:
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        dense = self._dense_search(qvec, DENSE_TOP_K)
        sparse = self._sparse_search(query, SPARSE_TOP_K)
        if timings is not None:
            timings["search_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        fused = self._rrf_fuse(dense, sparse)[:FUSED_TOP_K]
        candidate_idxs = [idx for idx, _ in fused]
        if timings is not None:
            timings["fuse_ms"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        reranked = self._rerank(query, candidate_idxs, RERANK_TOP_K)
        if timings is not None:
            timings["rerank_ms"] = (time.perf_counter() - t3) * 1000

        results = []
        for idx, score in reranked:
            c = self.chunks[idx]
            results.append(RetrievedChunk(
                chunk_id=c["chunk_id"], doc_id=c["doc_id"],
                title=c["title"], text=c["text"], score=score,
            ))
        return results

    def centroid_similarity(self, query_vec: np.ndarray) -> float:
        from app.config import CENTROID_PATH
        centroid = np.load(CENTROID_PATH)
        return float(np.dot(query_vec, centroid))


@lru_cache(maxsize=1)
def get_index() -> RetrievalIndex:
    """Process-wide singleton — loaded once, reused by every request."""
    return RetrievalIndex()
