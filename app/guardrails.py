"""
Two guardrails:

  input_guardrail  — runs BEFORE retrieval. Rejects queries that are nowhere
                      near the corpus topic, so we don't waste the retrieval/
                      generation budget (or hallucinate an answer) on
                      off-topic questions.

  output_guardrail — runs AFTER generation. Cheap lexical-overlap check that
                      each generated sentence actually overlaps with the
                      chunk it's supposedly grounded in. Flags ungrounded
                      claims rather than silently shipping them.
"""
import re
import numpy as np
from app.config import CENTROID_SIM_THRESHOLD, GROUNDEDNESS_MIN_OVERLAP

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def input_guardrail(query_vec: np.ndarray, centroid: np.ndarray) -> tuple[bool, float]:
    """Returns (is_on_topic, similarity). Caller decides what to say when False."""
    sim = float(np.dot(query_vec, centroid))
    return sim >= CENTROID_SIM_THRESHOLD, sim


def _sentence_split(text: str) -> list[str]:
    # good enough for short generated answers; not meant for arbitrary prose
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def output_guardrail(answer: str, context_chunks: list[str]) -> dict:
    """
    For each sentence in the answer, checks word-overlap against the best
    matching context chunk. Returns a report; does not silently rewrite the
    answer — the caller decides whether to append a caveat, strip the
    sentence, or log it for review.
    """
    context_word_sets = [_words(c) for c in context_chunks]
    sentences = _sentence_split(answer)
    report = {"sentences": [], "overall_grounded": True}

    for sent in sentences:
        sw = _words(sent)
        if not sw:
            continue
        best_overlap = 0.0
        for cw in context_word_sets:
            if not cw:
                continue
            overlap = len(sw & cw) / len(sw)
            best_overlap = max(best_overlap, overlap)
        grounded = best_overlap >= GROUNDEDNESS_MIN_OVERLAP
        report["sentences"].append({
            "text": sent, "overlap": round(best_overlap, 3), "grounded": grounded
        })
        if not grounded:
            report["overall_grounded"] = False

    return report
