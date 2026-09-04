"""
Central configuration module for the Local RAG System.
Handles environment variables, offline / confidential guarantees,
model selection, RAM mitigation settings, and paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# --- Confidentiality / Security Isolation ---
# Prevent any telemetry or unexpected network requests to Hugging Face / external clouds
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent

# Load .env if present
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# --- External Service Credentials ---
GITHUB_PAT: str = os.getenv("GITHUB_PAT", "")
ADO_PAT: str = os.getenv("ADO_PAT", "")
ADO_ORGANIZATION: str = os.getenv("ADO_ORGANIZATION", "")
ADO_PROJECT: str = os.getenv("ADO_PROJECT", "")

CONFLUENCE_URL: str = os.getenv("CONFLUENCE_URL", "").rstrip("/")
CONFLUENCE_EMAIL: str = os.getenv("CONFLUENCE_EMAIL", "")
CONFLUENCE_API_TOKEN: str = os.getenv("CONFLUENCE_API_TOKEN", "")

# --- Ollama Local Model Configuration ---
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")

# LLM Selection & Low-RAM Fallback
DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "llama3.1:8b")
FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "llama3.2:3b")
LOW_RAM_MODE: bool = os.getenv("LOW_RAM_MODE", "false").lower() in ("true", "1", "yes")

def get_active_llm_model() -> str:
    """Returns the LLM model to use based on RAM mode."""
    return FALLBACK_LLM_MODEL if LOW_RAM_MODE else DEFAULT_LLM_MODEL

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM: int = 768

# Batch size for embedding calls to prevent memory spikes in Ollama / RAM
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# --- Reranker Configuration ---
# Uses FlashRank by default (CPU ONNX, <100MB RAM, zero GPU dependency)
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "flashrank")

# --- Storage Configuration ---
DUCKDB_PATH: Path = Path(os.getenv("DUCKDB_PATH", str(BASE_DIR / "data" / "duckdb" / "store.duckdb")))
REPOS_CACHE_DIR: Path = Path(os.getenv("REPOS_CACHE_DIR", str(BASE_DIR / "data" / "repos")))
MAX_DUCKDB_MEMORY: str = os.getenv("MAX_DUCKDB_MEMORY", "2GB")

# Ensure required local directories exist
DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
REPOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Chunking & Token Budget Parameters ---
# Budgeted with headroom for nomic-embed-text / llama context windows
CHUNK_TARGET_TOKENS: int = int(os.getenv("CHUNK_TARGET_TOKENS", "450"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "768"))

# --- Retrieval Parameters ---
TOP_K_HYBRID: int = int(os.getenv("TOP_K_HYBRID", "25"))
TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", "5"))
RRF_K: int = int(os.getenv("RRF_K", "60"))
