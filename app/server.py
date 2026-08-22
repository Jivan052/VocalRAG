"""
FastAPI server.

Endpoints:
  GET  /                 -> serves static/index.html (the mic UI)
  POST /api/voice-query   -> multipart audio upload -> full voice pipeline
  POST /api/text-query     -> {"query": "..."} -> same pipeline minus STT (for testing)
  GET  /api/health         -> readiness check (confirms index is loaded)

Run with:
    uvicorn app.server:app --reload --port 8000
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dataclasses import asdict

from app.orchestration import run_voice_query, run_text_query
from app.retrieval import get_index
from app.config import ROOT_DIR

app = FastAPI(title="Voice RAG Assistant")


class TextQuery(BaseModel):
    query: str


@app.on_event("startup")
def _load_index_on_startup():
    # Fail fast at startup rather than on the first request if the index
    # hasn't been built yet.
    get_index()


@app.get("/api/health")
def health():
    idx = get_index()
    return {"status": "ok", "chunks_indexed": len(idx.chunks)}


@app.post("/api/voice-query")
async def voice_query(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload")
    result = await run_voice_query(
        audio_bytes, audio.filename or "audio.webm",
        audio.content_type or "audio/webm",
    )
    if result.stage == "stt_failed":
        raise HTTPException(status_code=503, detail=result.error or "Speech-to-text failed")
    return asdict(result)


@app.post("/api/text-query")
async def text_query(body: TextQuery):
    if not body.query.strip():
        raise HTTPException(400, "Empty query")
    result = await run_text_query(body.query)
    return asdict(result)


# --- static frontend ---
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")


@app.get("/")
def index_page():
    return FileResponse(str(ROOT_DIR / "static" / "index.html"))
