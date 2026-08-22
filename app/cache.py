"""
Exact-match + semantic cache on the normalized query.

In-memory, single-process. Good enough for a single voice-assistant instance;
swap the dict for Redis if you scale to multiple worker processes and want
cache hits to be shared across them.
"""
import re
import numpy as np
from app.config import SEMANTIC_CACHE_SIM_THRESHOLD, CACHE_MAX_ENTRIES

_WHITESPACE = re.compile(r"\s+")


def normalize(query: str) -> str:
    q = query.lower().strip()
    q = _WHITESPACE.sub(" ", q)
    q = q.rstrip("?.! ")
    return q


class SemanticCache:
    def __init__(self):
        self._exact: dict[str, dict] = {}
        self._vectors: list[np.ndarray] = []
        self._keys: list[str] = []

    def get(self, query: str, query_vec: np.ndarray) -> dict | None:
        key = normalize(query)
        if key in self._exact:
            return self._exact[key]

        if not self._vectors:
            return None
        sims = np.dot(np.stack(self._vectors), query_vec)
        best_i = int(np.argmax(sims))
        if sims[best_i] >= SEMANTIC_CACHE_SIM_THRESHOLD:
            return self._exact[self._keys[best_i]]
        return None

    def set(self, query: str, query_vec: np.ndarray, value: dict):
        key = normalize(query)
        if key in self._exact:
            self._exact[key] = value
            return
        if len(self._keys) >= CACHE_MAX_ENTRIES:
            oldest = self._keys.pop(0)
            self._vectors.pop(0)
            self._exact.pop(oldest, None)
        self._exact[key] = value
        self._keys.append(key)
        self._vectors.append(query_vec)


_cache = SemanticCache()


def get_cache() -> SemanticCache:
    return _cache
