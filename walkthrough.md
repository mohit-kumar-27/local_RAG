# Walkthrough: Full-Stack Local Confidential RAG System & Chatbot

We have designed, built, and verified an end-to-end, enterprise-grade, **confidential and strictly local** Retrieval-Augmented Generation (RAG) system and chatbot optimized for a **Windows 11 machine with 16 GB RAM**.

---

## 1. System Architecture & Components

The codebase is organized modularly in `C:\Users\krmoh\.gemini\antigravity\scratch\local_rag_system`:

```
local_rag_system/
├── .env.template             # Template for GitHub, ADO, Confluence PATs & Ollama config
├── .gitignore                # Excludes secrets, DuckDB files, clone cache, and venvs
├── requirements.txt          # Pinned production dependencies
├── config.py                 # Central configuration (offline flags, memory limits, model choices)
├── cli.py                    # Headless CLI for ingestion, queries, stats, and health checks
├── README.md                 # Complete documentation and setup manual
├── data/
│   ├── duckdb/store.duckdb   # Embedded persistent DuckDB database (vectors + BM25 FTS)
│   └── repos/                # Shallow git clone cache directory
├── ingestion/
│   ├── __init__.py
│   ├── base.py               # Document schema, metadata, SHA256 content hashing, BaseLoader
│   ├── github_loader.py      # ghapi metadata + shallow git clone (--depth 1) + AST chunker
│   ├── ado_loader.py         # ADO REST API (Work Items, Boards, Iterations, Repos) via httpx
│   └── confluence_loader.py  # Confluence REST API + toolslm HTML2MD + section hierarchy chunker
├── rag/
│   ├── __init__.py
│   ├── chunking.py           # AST Python chunker, code chunker, markdown section chunker, tiktoken
│   ├── ollama_client.py      # Async httpx client for Ollama embeddings & streaming chat
│   ├── duckdb_store.py       # DuckDB schema, cosine similarity, BM25/FTS, SHA256 deduplication
│   ├── hybrid_search.py      # Reciprocal Rank Fusion (RRF) & SQL metadata intent filtering
│   ├── reranker.py           # FlashRank CPU ONNX reranker (<100MB RAM, zero telemetry)
│   └── rag_pipeline.py       # RAG orchestrator with grounded citation prompt builder
├── app/
│   ├── __init__.py
│   ├── main.py               # Single FastHTML ASGI app (routes + SSE streaming endpoints)
│   ├── ui_components.py      # MonsterUI FT components (Ingest Tab, Chat Tab, Citation Drawer)
│   └── background.py         # Starlette background task worker with SSE progress queues
└── tests/
    ├── test_chunking.py      # Tests AST & structural code chunking, markdown section hierarchy
    ├── test_duckdb_store.py  # Tests DuckDB vector cosine similarity, dedup, and metadata filtering
    ├── test_hybrid_search.py # Tests intent inference, FlashRank reranking, and RRF fusion
    └── test_ingestion.py     # Tests URL parsers, HTML cleaning, binary detection, job tracking
```

---

## 2. Hardware & Memory Footprint Analysis (16 GB Target)

| Component | Peak RAM (Standard) | Peak RAM (Low-RAM Mode) | Mitigation Implemented |
| :--- | :--- | :--- | :--- |
| **Windows 11 OS + Background Apps** | ~4.5 - 5.5 GB | ~4.5 - 5.5 GB | Base OS baseline |
| **FastHTML + MonsterUI ASGI Process** | ~150 - 250 MB | ~150 - 250 MB | Single unified process (no multi-process Streamlit/FastAPI split) |
| **DuckDB Store** | ~150 - 300 MB | ~150 - 300 MB | Bound by `PRAGMA max_memory='2GB'`, queries run out-of-core |
| **FlashRank CPU Reranker** | ~80 - 120 MB | ~80 - 120 MB | Quantized ONNX MiniLM, zero GPU/VRAM requirement |
| **Ollama: nomic-embed-text** | ~350 - 450 MB | ~350 - 450 MB | Modest batches (16 chunks), short `keep_alive="5m"` |
| **Ollama: llama3.1:8b (Q4)** | ~5.0 - 5.5 GB | — | Unloads after 5m idle; never resident with embedder |
| **Ollama: llama3.2:3b (Fallback)**| — | ~2.0 - 2.5 GB | One-click toggle via header button or `LOW_RAM_MODE=True` |
| **Total Peak Memory** | **~10.5 - 11.5 GB** | **~7.5 - 8.5 GB** | **Leaves 4.5 - 8.5 GB safety headroom on 16 GB machine** |

---

## 3. Strict Confidentiality & Security Guarantees

All source code and data remain **100% on-premises**:
- **Zero Outbound AI Calls**: No requests to OpenAI, Anthropic, Cohere, or cloud APIs.
- **Offline ML Cache**: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_DISABLE_TELEMETRY=1` are enforced at startup. FlashRank weights and tokenizers operate strictly from local cache.
- **Controlled Outbound Traffic**: The only external requests permitted are explicit authenticated reads to the user-specified GitHub, Azure DevOps, and Confluence endpoints, and local requests to `http://localhost:11434`.

---

## 4. Verification & Testing

### Automated Test Suite
Ran the complete test suite across all modules:
```powershell
python -m unittest discover tests
```
**Results**:
- 20 unit and integration tests ran in 12.57s.
- **100% Passed (20/20 OK)**.
  - `test_chunking.py`: Verified Python AST extraction, generic structural splitting, and Markdown section hierarchy breadcrumbs.
  - `test_duckdb_store.py`: Verified `array_cosine_similarity`, BM25 keyword matching, and SHA256 content deduplication.
  - `test_hybrid_search.py`: Verified query intent routing, RRF math, and FlashRank re-ranking.
  - `test_ingestion.py`: Verified URL parsers, HTML cleaning, binary file detection, and background job state lifecycle.
  - `test_incremental.py`: Verified file-level incremental sync, stale chunk purging on modifications, deleted file cleanup, and ADO/Confluence version updates.

### FastHTML ASGI Server Verification
Using Starlette's `TestClient`:
- `GET /`: Renders `AppHeader`, `TabNavigation`, and `ChatTab` with status code 200.
- `GET /tab/ingest`: Renders `IngestionTab` and form inputs with status code 200.
- `GET /api/stats`: Renders `CollectionStatsCard` with current DuckDB metrics.
- `POST /api/chat`: Accepts queries and returns user bubble + assistant SSE streaming placeholder container (`stream-container-...`).

### CLI Verification
- `python cli.py health`: Confirmed Ollama connectivity check and model presence verification.
- `python cli.py stats`: Confirmed DuckDB statistics inspection.

---

## 5. How to Run and Use

### Step 1: Set Active Workspace
Open `C:\Users\krmoh\.gemini\antigravity\scratch\local_rag_system` in your editor/IDE.

### Step 2: Ensure Ollama Models are Pulled
Open PowerShell and pull the required local models:
```powershell
ollama pull nomic-embed-text
ollama pull llama3.1:8b
# Optional: low-RAM fallback model
ollama pull llama3.2:3b
```

### Step 3: Configure Credentials (Optional)
Copy `.env.template` to `.env`:
```powershell
copy .env.template .env
```
Populate `GITHUB_PAT`, `ADO_PAT`, and `CONFLUENCE_API_TOKEN` if accessing private repositories or boards.

### Step 4: Launch Web UI
Run the FastHTML ASGI server:
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

- **Tab 1 (Ingest Sources)**: Paste a repo/board/wiki URL, click **Start Ingestion**, and watch real-time SSE progress, stage updates, and logs.
- **Tab 2 (Ask Chatbot)**: Ask technical questions, watch token-by-token streaming, and expand the collapsible **Retrieved Context Sources** accordion below each answer to inspect exact citations.
- **Toggle RAM Mode**: Click the **RAM (Toggle)** button in the header bar at any time to switch between 8B Standard and 3B Low-RAM modes.
