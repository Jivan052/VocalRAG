"""
Generation. Talks to OpenRouter's OpenAI-compatible /chat/completions
endpoint by default, so a single OPENROUTER_API_KEY covers whatever model
you point GENERATION_MODEL at. This sits outside the 200ms retrieval budget
by design — it's a network-bound call, same as STT.

Set GENERATION_PROVIDER = "anthropic" in app/config.py (and export
ANTHROPIC_API_KEY) if you'd rather call Anthropic directly instead.
"""
import httpx
from app.config import (
    GENERATION_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    GENERATION_MODEL, GENERATION_MAX_TOKENS, ANTHROPIC_API_KEY,
)

SYSTEM_PROMPT = (
    "You are a voice assistant. Answer ONLY using the provided context chunks. "
    "If the context doesn't contain the answer, say you don't have that "
    "information rather than guessing. Keep answers to 2-3 short sentences — "
    "this will be read aloud, so avoid lists, markdown, or long asides."
)


class GenerationError(RuntimeError):
    pass


def build_prompt(query: str, chunks: list[str]) -> str:
    context_block = "\n\n".join(f"[chunk {i+1}] {c}" for i, c in enumerate(chunks))
    return f"Context:\n{context_block}\n\nQuestion: {query}"


def _generate_openrouter(query: str, chunks: list[str]) -> str:
    if not OPENROUTER_API_KEY:
        raise GenerationError(
            "OPENROUTER_API_KEY is not set. Add it to .env and restart the server."
        )
    prompt = build_prompt(query, chunks)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for attribution/rate-limit tiers:
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Voice RAG Assistant",
    }
    body = {
        "model": GENERATION_MODEL,
        "max_tokens": GENERATION_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{OPENROUTER_BASE_URL}/chat/completions",
                            headers=headers, json=body)
    if resp.status_code != 200:
        raise GenerationError(f"OpenRouter failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise GenerationError(f"Unexpected OpenRouter response shape: {data}") from e


def _generate_anthropic(query: str, chunks: list[str]) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = build_prompt(query, chunks)
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=GENERATION_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def generate_answer(query: str, chunks: list[str]) -> str:
    if GENERATION_PROVIDER == "anthropic":
        return _generate_anthropic(query, chunks)
    return _generate_openrouter(query, chunks)


async def _generate_openrouter_stream(query: str, chunks: list[str]):
    """Async generator yielding text deltas via OpenRouter's SSE stream."""
    import json as _json

    if not OPENROUTER_API_KEY:
        raise GenerationError("OPENROUTER_API_KEY is not set.")
    prompt = build_prompt(query, chunks)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Voice RAG Assistant",
    }
    body = {
        "model": GENERATION_MODEL,
        "max_tokens": GENERATION_MAX_TOKENS,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", f"{OPENROUTER_BASE_URL}/chat/completions",
                                  headers=headers, json=body) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = _json.loads(payload)["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, ValueError):
                    continue
                if delta:
                    yield delta


async def generate_answer_stream(query: str, chunks: list[str]):
    """Streaming variant. Only wired up for OpenRouter for now — call
    generate_answer() for the Anthropic-direct path."""
    async for delta in _generate_openrouter_stream(query, chunks):
        yield delta
