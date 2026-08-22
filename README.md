# Voice RAG Assistant

Voice-in, grounded-answer-out. Hybrid retrieval (FAISS + BM25, both
in-process, no vector DB service) sits under a 200ms budget; STT and
generation are network calls outside that budget by design.

```
mic → ElevenLabs STT → input guardrail → cache check
    → dense search (FAISS) ┐
    → sparse search (BM25) ┴→ RRF fuse → cross-encoder rerank
    → LLM generation → output guardrail (groundedness) → answer
```

## 1. Install

```bash
cd voice-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Set API keys

Copy `.env.example` to `.env` and fill in the keys. The application loads
`.env` automatically from the project root, and `.env` is ignored by git.

Generation defaults to **OpenRouter** (one key, any model on it):

```bash
cp .env.example .env
# edit .env and set ELEVENLABS_API_KEY and OPENROUTER_API_KEY
```

To call Anthropic directly instead, set `GENERATION_PROVIDER=anthropic` and
`ANTHROPIC_API_KEY` in `.env`.

## 3. Build the index (run once, and again whenever data/corpus.json changes)

```bash
python build_index.py
```

This writes `index/dense.faiss`, `index/bm25.pkl`, `index/chunks.json`,
`index/centroid.npy`. Ships with a 6-doc sample corpus (return policy,
tracking, payments, warranty, account help, cancellations) so the whole
pipeline runs end to end immediately.

## 4. Run the server

```bash
uvicorn app.server:app --reload --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 — tap the mic, ask a question, watch the signal
path panel show per-stage latency (matches your retrieval-budget screenshot).

Or test without a mic:
```bash
curl -X POST localhost:8000/api/text-query -H 'Content-Type: application/json' \
  -d '{"query": "how long do I have to return something"}'
```

### Deploy on Render

The included `render.yaml`, `Procfile`, and `runtime.txt` configure the web
service to bind to Render's `$PORT` and use Python 3.12. Add
`OPENROUTER_API_KEY` and `ELEVENLABS_API_KEY` as secret environment variables
in the Render dashboard, then deploy the repository.

## Swapping in MSMARCO-XI (or any real corpus)

Replace `data/corpus.json` with a JSON array of
`{"id": ..., "title": ..., "text": ...}` objects, then rerun
`python build_index.py`. Nothing else changes — chunking, embedding,
indexing, and retrieval all key off that one file.

## Where each architecture decision lives

| Decision | File |
|---|---|
| Chunking strategy (fixed+overlap; semantic/parent-child stubbed) | `app/chunking.py` |
| Embedding model | `app/embedding.py`, `EMBEDDING_MODEL` in `app/config.py` |
| Dense index (FAISS HNSW, in-process) | `build_index.py`, `app/retrieval.py` |
| Sparse index (BM25) | `build_index.py`, `app/retrieval.py` |
| RRF fusion + cross-encoder rerank | `app/retrieval.py` |
| Exact + semantic cache | `app/cache.py` |
| Input guardrail (centroid similarity) | `app/guardrails.py` |
| Output guardrail (groundedness/overlap) | `app/guardrails.py` |
| Generation (OpenRouter by default, capped tokens, streamed variant available) | `app/generation.py` |
| ElevenLabs STT | `app/stt.py` |
| Orchestration / per-stage timing / retries | `app/orchestration.py` |
| Tunable numbers (top-k, thresholds, budget) | `app/config.py` |

## Notes

- First run downloads the MiniLM embedding model and the cross-encoder
  reranker (~100MB total) from HuggingFace — needs network the first time,
  then it's cached locally.
- The retrieval budget panel in the UI sums `query_embed_ms` +
  `retrieval.*` + `cache_lookup_ms` and compares it to
  `RETRIEVAL_BUDGET_MS` (200ms) client-side.
- `run_text_query` in `app/orchestration.py` shares 100% of the logic with
  `run_voice_query` — only the STT step differs — so text-only testing is a
  true test of the retrieval/generation path.
