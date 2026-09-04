# Local Confidential RAG System & Chatbot

An enterprise-grade, **strictly confidential and local** Retrieval-Augmented Generation (RAG) system and chatbot designed for **Windows 11 with 16 GB RAM**. 

Built with the **answer.ai / fast.ai** stack: **FastHTML**, **MonsterUI**, **DuckDB**, **Ollama**, **FlashRank**, **ghapi**, **fastcore**, and **toolslm**.

---

## Key Features

1. **100% Local & Confidential Execution**:
   - Zero outbound telemetry or cloud AI calls (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`).
   - All vectors and metadata are stored in an embedded DuckDB file (`./data/duckdb/store.duckdb`).
   - Only local models via Ollama (`http://localhost:11434`) are used for embeddings and generation.
2. **Multi-Source Connectors**:
   - **GitHub Repositories**: Authenticated access via `ghapi` (`GhApi`) for issues/PRs/metadata, coupled with **shallow `git clone --depth 1`** for high-throughput, rate-limit-free bulk source ingestion.
   - **Azure DevOps**: REST API (`httpx.AsyncClient`) extracting Work Items (Epics, Stories, Bugs, Tasks), iteration paths (Sprints), acceptance criteria, and discussion comments, plus ADO Git repositories.
   - **Atlassian Confluence**: REST API (`httpx.AsyncClient`) with `toolslm` HTML-to-Markdown conversion and **hierarchy-preserving section boundary chunking**.
3. **Language-Aware & Hierarchy-Preserving Chunking**:
   - **Python**: AST-based code chunking preserving function/class boundaries, decorators, signatures, and docstrings.
   - **Multi-Language (TypeScript, Go, C#, Java)**: Structural line-based chunking with token budgeting (`tiktoken`) and context headers.
   - **Confluence / Docs**: Header-hierarchy breadcrumbs (`# Title > ## Section > ### Subsection`) keeping full context in every chunk.
   - **True Incremental Sync & Deduplication**:
     - **Git Diffs**: Ingestion inspects `git diff --name-status old_head new_head` on `git pull` to isolate Added, Modified, and Deleted files.
     - **Stale Chunk Pruning**: Modifying or deleting a file automatically purges its obsolete vector chunks from DuckDB (`delete_documents_by_file`).
     - **SHA256 Deduplication**: Every chunk carries a deterministic content hash. Chunks that haven't changed skip Ollama embedding generation entirely, saving massive CPU/RAM and time.
     - **Work Item & Wiki Sync**: Updates to ADO tickets or Confluence pages purge earlier versions (`delete_documents_by_work_item`) before indexing fresh state.
4. **Hybrid Retrieval & Local Re-Ranking**:
   - **Dense Retrieval**: DuckDB native vector cosine similarity (`array_cosine_similarity` over `FLOAT[768]`).
   - **Sparse Retrieval**: DuckDB Full-Text Search (BM25 FTS extension) with keyword fallback.
   - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings with $k=60$.
   - **FlashRank Re-ranking**: Ultra-lightweight CPU ONNX cross-encoder (<100 MB RAM).
5. **Memory-First Design (16 GB Windows 11 Profile)**:
   - Single FastHTML ASGI process (no separate Streamlit or multi-tier servers).
   - Short Ollama `keep_alive` (`5m`) ensures idle models unload from RAM promptly.
   - One-click Low-RAM Mode fallback (`llama3.2:3b` / `qwen2.5-coder:3b`) when memory headroom is constrained.
   - DuckDB memory bounded via `PRAGMA max_memory='2GB'`.
6. **Streaming UI (FastHTML + MonsterUI)**:
   - Tab 1: **Ingest Sources** — Live SSE progress bar, status stages, logs, and document collection statistics.
   - Tab 2: **Ask Chatbot** — Token-by-token streaming chat with a collapsible MonsterUI citation drawer showing exact file paths, line ranges, work item IDs, and relevance scores.
7. **CLI Interface**:
   - Headless scripting support for ingestion, querying, collection stats, and health checks.

---

## Hardware & Memory Budget Profile (16 GB Machine)

| Component | Peak RAM (Standard) | Peak RAM (Low-RAM Mode) | Notes |
| :--- | :--- | :--- | :--- |
| **Windows 11 OS & Base Apps** | ~4.5 - 5.5 GB | ~4.5 - 5.5 GB | System baseline |
| **FastHTML / MonsterUI ASGI App** | ~150 - 250 MB | ~150 - 250 MB | Single unified process |
| **DuckDB Store** | ~150 - 300 MB | ~150 - 300 MB | Bounded by `PRAGMA max_memory='2GB'` |
| **FlashRank CPU Reranker** | ~80 - 120 MB | ~80 - 120 MB | Quantized ONNX MiniLM |
| **Ollama: nomic-embed-text** | ~350 - 450 MB | ~350 - 450 MB | Batched (16 chunks), 5m keep-alive |
| **Ollama: LLM (`llama3.1:8b`)** | ~5.0 - 5.5 GB | — | Default model (Q4_K_M) |
| **Ollama: LLM (`llama3.2:3b`)** | — | ~2.0 - 2.5 GB | Fallback mode (Q4_K_M) |
| **Total System RAM** | **~10.5 - 11.5 GB** | **~7.5 - 8.5 GB** | **Leaves 4.5 - 8.5 GB safety buffer** |

---

## Quick Start Guide

### 1. Prerequisites

- Windows 11 with Python 3.12+ and Git installed.
- [Ollama](https://ollama.com/) running locally on `http://localhost:11434`.

Pull the required models:
```powershell
# Required embedding model (768 dimensions)
ollama pull nomic-embed-text

# Standard LLM (Default)
ollama pull llama3.1:8b

# Low-RAM fallback LLM (Optional, for <12GB free RAM)
ollama pull llama3.2:3b
```

### 2. Environment Configuration

Copy `.env.template` to `.env` and set your credentials:
```powershell
copy .env.template .env
```

Edit `.env` as needed:
- `GITHUB_PAT`: Personal Access Token with repo read scope.
- `ADO_PAT`, `ADO_ORGANIZATION`, `ADO_PROJECT`: Azure DevOps credentials.
- `CONFLUENCE_URL`, `CONFLUENCE_EMAIL`, `CONFLUENCE_API_TOKEN`: Atlassian Confluence credentials.
- `LOW_RAM_MODE`: Set to `true` to use `llama3.2:3b` instead of `llama3.1:8b`.

### 3. Verification & CLI Usage

Check system connectivity and model status:
```powershell
python cli.py health
```

View indexed knowledge base stats:
```powershell
python cli.py stats
```

Ingest a repository from the command line:
```powershell
python cli.py ingest --type github --url https://github.com/fastai/fastcore
```

Ask a question with citations:
```powershell
python cli.py query "How does function dispatching work in fastcore?" --filter code
```

### 4. Running the Web Application

Launch the FastHTML ASGI application:
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

- **Tab 1 (Ingest Sources)**: Paste your repo or board URL, start ingestion, and monitor real-time SSE progress.
- **Tab 2 (Ask Chatbot)**: Conversational chat with token-by-token streaming and collapsible citation drawers below every answer.
- **Toggle RAM Mode**: Click the **RAM (Toggle)** button in the top right header to instantly switch between 8B Standard and 3B Low-RAM modes.

### 5. Running Automated Tests

Run the full unit and integration test suite:
```powershell
python -m unittest discover tests
```
