"""
Orchestration: a small typed pipeline, not a heavyweight framework, since the
flow is linear with one branch point (cache hit / guardrail reject).

Each stage records its own timing into a shared dict so the client can render
the same kind of latency breakdown as your screenshot panel. Stages that call
an external network API (STT, generation) get a basic retry; local stages
(embedding, retrieval, guardrails) don't need one — if they fail, retrying
in-process won't help.
"""
import time
from dataclasses import dataclass, field

from app.config import RERANK_TOP_K
from app.stt import transcribe_audio, STTError
from app.embedding import embed_query
from app.retrieval import get_index
from app.guardrails import input_guardrail, output_guardrail
from app.generation import generate_answer
from app.cache import get_cache


@dataclass
class PipelineResult:
    answer: str
    query_text: str
    stage: str  # which stage the pipeline stopped/succeeded at
    citations: list[dict] = field(default_factory=list)
    groundedness: dict | None = None
    timings_ms: dict = field(default_factory=dict)
    cache_hit: bool = False
    on_topic: bool = True
    error: str | None = None


async def _with_retry(fn, *args, retries=1, **kwargs):
    last_err = None
    for attempt in range(retries + 1):
        try:
            return await fn(*args, **kwargs) if _is_coro(fn) else fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — deliberately broad, we log+retry
            last_err = e
            if attempt < retries:
                continue
    raise last_err


def _is_coro(fn):
    import inspect
    return inspect.iscoroutinefunction(fn)


async def run_voice_query(audio_bytes: bytes, filename: str, content_type: str) -> PipelineResult:
    timings: dict = {}
    t_total = time.perf_counter()

    # 1. Speech to text (network call, retried once)
    t0 = time.perf_counter()
    try:
        query_text = await _with_retry(
            transcribe_audio, audio_bytes, filename, content_type, retries=1
        )
    except STTError as e:
        return PipelineResult(
            answer="Sorry, I couldn't hear that clearly — could you try again?",
            query_text="", stage="stt_failed", error=str(e),
            timings_ms={"stt_ms": (time.perf_counter() - t0) * 1000},
        )
    timings["stt_ms"] = (time.perf_counter() - t0) * 1000

    return await run_text_query(query_text, timings, t_total)


async def run_text_query(query_text: str, timings: dict | None = None,
                          _t_total: float | None = None) -> PipelineResult:
    """Split out from run_voice_query so the text-only /api/query endpoint
    (useful for testing without a microphone) shares the exact same logic."""
    timings = timings if timings is not None else {}
    t_total = _t_total if _t_total is not None else time.perf_counter()

    index = get_index()
    cache = get_cache()

    # 2. Embed once, reuse for guardrail + cache + retrieval
    t1 = time.perf_counter()
    qvec = embed_query(query_text)
    timings["query_embed_ms"] = (time.perf_counter() - t1) * 1000

    # 3. Input guardrail — off-topic short circuit
    import numpy as np
    from app.config import CENTROID_PATH
    centroid = np.load(CENTROID_PATH)
    on_topic, sim = input_guardrail(qvec, centroid)
    if not on_topic:
        return PipelineResult(
            answer="That's outside what I can help with here — I can only "
                   "answer questions about this product/service's documentation.",
            query_text=query_text, stage="input_guardrail_rejected",
            on_topic=False, timings_ms=timings,
        )

    # 4. Cache check
    t2 = time.perf_counter()
    cached = cache.get(query_text, qvec)
    timings["cache_lookup_ms"] = (time.perf_counter() - t2) * 1000
    if cached:
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        return PipelineResult(
            answer=cached["answer"], query_text=query_text, stage="cache_hit",
            citations=cached["citations"], groundedness=cached["groundedness"],
            cache_hit=True, timings_ms=timings,
        )

    # 5. Hybrid retrieval (dense + sparse -> RRF fuse -> cross-encoder rerank)
    retrieval_timings: dict = {}
    chunks = index.retrieve(query_text, timings=retrieval_timings)
    timings.update({f"retrieval.{k}": v for k, v in retrieval_timings.items()})

    if not chunks:
        return PipelineResult(
            answer="I don't have information on that.",
            query_text=query_text, stage="no_chunks_found", timings_ms=timings,
        )

    # 6. Generation
    t3 = time.perf_counter()
    context_texts = [c.text for c in chunks[:RERANK_TOP_K]]
    answer = generate_answer(query_text, context_texts)
    timings["generation_ms"] = (time.perf_counter() - t3) * 1000

    # 7. Output guardrail
    t4 = time.perf_counter()
    groundedness = output_guardrail(answer, context_texts)
    timings["output_guardrail_ms"] = (time.perf_counter() - t4) * 1000

    citations = [
        {"chunk_id": c.chunk_id, "title": c.title, "score": round(c.score, 4)}
        for c in chunks
    ]

    result = {"answer": answer, "citations": citations, "groundedness": groundedness}
    cache.set(query_text, qvec, result)

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000
    return PipelineResult(
        answer=answer, query_text=query_text, stage="generated",
        citations=citations, groundedness=groundedness, timings_ms=timings,
    )
