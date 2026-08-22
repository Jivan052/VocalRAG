"""
ElevenLabs Speech-to-Text wrapper.

This is a network-bound API call, so it sits outside the 200ms retrieval
budget — its latency is whatever ElevenLabs + your network gives you, and
that's expected/fine per the architecture split.
"""
import httpx
from app.config import ELEVENLABS_API_KEY, ELEVENLABS_STT_URL, ELEVENLABS_STT_MODEL


class STTError(RuntimeError):
    pass


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm",
                            content_type: str = "audio/webm") -> str:
    if not ELEVENLABS_API_KEY:
        raise STTError(
            "ELEVENLABS_API_KEY is not set. Add it to .env and restart the server."
        )

    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    files = {"file": (filename, audio_bytes, content_type)}
    data = {"model_id": ELEVENLABS_STT_MODEL}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            ELEVENLABS_STT_URL, headers=headers, files=files, data=data
        )

    if resp.status_code != 200:
        raise STTError(f"ElevenLabs STT failed ({resp.status_code}): {resp.text[:300]}")

    payload = resp.json()
    text = payload.get("text", "")
    if not text:
        raise STTError("ElevenLabs STT returned no text — audio may be silent/unclear.")
    return text.strip()
