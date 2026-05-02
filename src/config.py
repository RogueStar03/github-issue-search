from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

CHROMA_PATH = str(BASE_DIR / "chroma_db")
BM25_PATH = BASE_DIR / "bm25_index"
STATE_PATH = BASE_DIR / "state"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"
EMBED_BATCH_SIZE = 32   # cap per ollama batch quality guidance
NUM_CTX = 8192          # qwen2.5:7b default is 2048 — too small for RAG
MAX_BODY_CHARS = 1500   # nomic-embed-text default context ~2048 tokens; 1500 chars ≈ 375 tokens

TOP_K_VECTOR = 50
TOP_K_BM25 = 50
TOP_K_FINAL = 10
RRF_K = 60