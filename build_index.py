"""
Offline indexing entrypoint.

Run this once (and again whenever data/corpus.json changes):
    python build_index.py

Builds three artifacts, all in-process / on-disk, no server required:
  - index/dense.faiss   FAISS HNSW index over chunk embeddings
  - index/bm25.pkl      BM25Okapi index over tokenized chunks
  - index/chunks.json   the chunk records themselves (id -> text/title/doc_id)
  - index/centroid.npy  mean corpus embedding, used by the input guardrail
"""
import json
import pickle
import time
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from app.config import (
    CORPUS_PATH, FAISS_INDEX_PATH, CHUNK_STORE_PATH, BM25_STORE_PATH,
    CENTROID_PATH, EMBEDDING_DIM, INDEX_DIR,
)
from app.chunking import build_chunks_from_corpus
from app.embedding import embed_texts


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def main():
    t0 = time.time()
    INDEX_DIR.mkdir(exist_ok=True)

    corpus = json.loads(CORPUS_PATH.read_text())
    print(f"Loaded {len(corpus)} source documents")

    chunks = build_chunks_from_corpus(corpus)
    print(f"Built {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)  # (N, dim), L2-normalized
    assert embeddings.shape[1] == EMBEDDING_DIM, (
        f"Embedding dim mismatch: got {embeddings.shape[1]}, "
        f"config says {EMBEDDING_DIM}. Update EMBEDDING_DIM in app/config.py."
    )

    # --- Dense index: HNSW over inner product (= cosine, since normalized) ---
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 64
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Wrote dense FAISS index -> {FAISS_INDEX_PATH}")

    # --- Sparse index: BM25 over the same chunks, same order/ids ---
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_STORE_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Wrote BM25 index -> {BM25_STORE_PATH}")

    # --- Chunk store (id -> metadata, in the same order as both indexes) ---
    CHUNK_STORE_PATH.write_text(json.dumps(chunks, indent=2))
    print(f"Wrote chunk store -> {CHUNK_STORE_PATH}")

    # --- Centroid for the input guardrail (off-topic query rejection) ---
    centroid = embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    np.save(CENTROID_PATH, centroid)
    print(f"Wrote corpus centroid -> {CENTROID_PATH}")

    print(f"Done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
