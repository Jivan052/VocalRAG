"""
Central configuration. Everything that's a "tune this later" knob lives here,
so the rest of the code never hardcodes a magic number.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "index"

load_dotenv(ROOT_DIR / ".env")

CORPUS_PATH = DATA_DIR / "corpus.json"
FAISS_INDEX_PATH = INDEX_DIR / "dense.faiss"
CHUNK_STORE_PATH = INDEX_DIR / "chunks.json"
BM25_STORE_PATH = INDEX_DIR / "bm25.pkl"
CENTROID_PATH = INDEX_DIR / "centroid.npy"

# --- Chunking (offline) ---
CHUNK_SIZE_CHARS = 400
CHUNK_OVERLAP_CHARS = 80

# --- Embedding ---
# Small local sentence-transformer. Swap for an ONNX/quantized export later
# for the last bit of latency without changing any calling code.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Retrieval ---
DENSE_TOP_K = 8
SPARSE_TOP_K = 8
RRF_K = 60          # reciprocal rank fusion constant
FUSED_TOP_K = 6      # how many candidates survive fusion, go to rerank
RERANK_TOP_K = 3     # how many survive rerank, go to the LLM as context

# --- Input guardrail ---
# Cosine similarity of the query embedding to the corpus centroid.
# Below this, the query is treated as off-topic and short-circuited
# before hitting retrieval/generation at all.
CENTROID_SIM_THRESHOLD = 0.15

# --- Cache ---
SEMANTIC_CACHE_SIM_THRESHOLD = 0.96
CACHE_MAX_ENTRIES = 500

# --- Generation ---
# Using OpenRouter (OpenAI-compatible endpoint) so you can point this at
# whatever model your OpenRouter key has access to, without an Anthropic key.
GENERATION_PROVIDER = os.environ.get("GENERATION_PROVIDER", "openrouter")  # "openrouter" or "anthropic"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Pick any model your OpenRouter key covers — small/fast keeps you inside
# the "generation is a network call, not the 200ms leg" assumption.
# Examples: "anthropic/claude-3.5-haiku", "openai/gpt-4o-mini", "google/gemini-2.0-flash-001"
GENERATION_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku")
GENERATION_MAX_TOKENS = int(os.environ.get("GENERATION_MAX_TOKENS", "220"))

# --- Output guardrail ---
# Minimum lexical/semantic overlap between a generated sentence and its
# cited chunk before we consider the claim "grounded."
GROUNDEDNESS_MIN_OVERLAP = 0.20

# --- STT: ElevenLabs ---
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_STT_URL = os.environ.get(
	"ELEVENLABS_STT_URL", "https://api.elevenlabs.io/v1/speech-to-text"
)
ELEVENLABS_STT_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")

# --- Anthropic (generation) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
