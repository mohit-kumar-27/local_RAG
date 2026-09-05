# Implementation Plan: Full-Stack Local RAG System & Chatbot

Build an end-to-end, enterprise-grade, **confidential and strictly local** Retrieval-Augmented Generation (RAG) system and chatbot on Windows 11 (16 GB RAM constraint). The system ingests GitHub repositories, Azure DevOps repos & boards, and Atlassian Confluence spaces, stores hybrid vector and keyword representations in DuckDB, and provides conversational answers with exact source citations via local Ollama models and FastHTML + MonsterUI.

## User Review Required

> [!IMPORTANT]
> **Local Ollama Models Required**:
> The system requires Ollama running on `http://localhost:11434` with the following models:
> 1. `nomic-embed-text` (768-dimensional embeddings)
> 2. `llama3.1:8b` (default LLM) or fallback `llama3.2:3b` / `qwen2.5-coder:3b` (for lower RAM usage)
>
> You can pull these via terminal:
> ```powershell
> ollama pull nomic-embed-text
> ollama pull llama3.1:8b
> # or low-RAM fallback:
> ollama pull llama3.2:3b
> ```

> [!NOTE]
> **Active Workspace Recommendation**:
> The project will be created in `C:\Users\krmoh\.gemini\antigravity\scratch\local_rag_system`. It is recommended to open or set this folder as your active workspace in your editor/IDE.

## Proposed Architecture & Directory Structure

```
C:\Users\krmoh\.gemini\antigravity\scratch\local_rag_system\
├── .env.template             # Template for GitHub, ADO, Confluence PATs & Ollama configs
├── .gitignore                # Excludes virtual environments, data, repos, cache
├── requirements.txt          # Pinned production dependencies
├── config.py                 # Central configuration (models, keep-alive, memory caps, paths)
├── cli.py                    # CLI interface for ingestion, queries, and store inspection
├── data/
│   ├── duckdb/store.duckdb   # Embedded persistent DuckDB database (vectors + FTS)
│   └── repos/                # Shallow git clone target directory for bulk ingestion
├── ingestion/
│   ├── __init__.py
│   ├── base.py               # Document schema, metadata, and abstract BaseLoader
│   ├── github_loader.py      # ghapi client + shallow git clone + AST/code chunker
│   ├── ado_loader.py         # ADO REST API (Work Items, Boards, Iterations, Repos)
│   └── confluence_loader.py  # Confluence REST API + Markdown conversion & section splitting
├── rag/
│   ├── __init__.py
│   ├── chunking.py           # AST Python chunker, code chunker, markdown section chunker + tiktoken
│   ├── ollama_client.py      # Async httpx client for Ollama embedding & chat streaming
│   ├── duckdb_store.py       # DuckDB schema, cosine similarity, BM25/FTS, SHA256 deduplication
│   ├── hybrid_search.py      # Reciprocal Rank Fusion (RRF) & SQL metadata intent filtering
│   ├── reranker.py           # FlashRank CPU ONNX reranker wrapper (zero telemetry, offline)
│   └── rag_pipeline.py       # RAG orchestrator, grounded prompt with inline citations
├── app/
│   ├── __init__.py
│   ├── main.py               # Single FastHTML ASGI application (routes + SSE streaming)
│   ├── ui_components.py      # MonsterUI components (Tab 1: Ingest, Tab 2: Chat & Citation Drawer)
│   └── background.py         # Starlette background task worker with SSE progress queues
└── tests/
    ├── test_chunking.py      # Unit tests for code & markdown hierarchy chunking
    ├── test_duckdb_store.py  # Unit tests for DuckDB vector similarity, FTS, and dedup
    ├── test_hybrid_search.py # Unit tests for RRF fusion and reranking
    └── test_ingestion.py     # Unit tests for loaders and error resilience
```

---

## Hardware & Memory Budget (16 GB Target on Windows 11)

| Subsystem / Process | Expected Peak RAM | Memory Mitigation Strategy |
| :--- | :--- | :--- |
| **Windows 11 OS + Background Apps** | ~4.5 - 5.5 GB | Base system baseline |
| **FastHTML / MonsterUI (ASGI Uvicorn)** | ~150 - 250 MB | Single unified process (no multi-tier Streamlit/FastAPI split) |
| **DuckDB Embedded Store** | ~200 - 500 MB | Capped with `PRAGMA max_memory='2GB'`, queries run out-of-core |
| **FlashRank Reranker (CPU ONNX)** | ~80 - 150 MB | TinyBERT/MiniLM ONNX quantized model; zero GPU overhead |
| **Ollama: nomic-embed-text** | ~350 - 450 MB | Short `keep_alive` (`5m`), chunk batching (16-32 chunks per call) |
| **Ollama: llama3.1:8b (Q4_K_M)** | ~4.9 - 5.5 GB | Unloads after 5m idle; never resident simultaneously with heavy embed batches |
| **Ollama: llama3.2:3b (Fallback Mode)** | ~2.0 - 2.5 GB | Toggleable via `config.py` (`LOW_RAM_MODE=True`) for constrained headroom |
| **Total Peak System RAM** | **~10.5 - 11.5 GB** *(Standard)* / **~7.5 - 8.5 GB** *(Low-RAM)* | **Leaves 4.5 - 8.5 GB safety buffer on 16 GB machine** |

---

## Step-by-Step Implementation Blueprint

### Phase 1: Environment, Configuration & Security Isolation
- **Dependencies**: Create `requirements.txt` with exact resolved versions:
  `python-fasthtml`, `monsterui`, `ghapi`, `httpx`, `duckdb`, `tiktoken`, `toolslm`, `fastcore`, `rerankers[flashrank]`, `uvicorn`.
- **Environment & Confidentiality**:
  - Create `.env.template` with all keys for GitHub PAT, ADO, Confluence, Ollama settings.
  - Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1` in `config.py` to prevent any outbound cloud calls.
- **Config**:
  - `config.py` managing Ollama endpoints, LLM model names, fallback switch (`LOW_RAM_MODE`), embedding dimension (768), DuckDB path, shallow clone cache directory, and chunk token limits.

### Phase 2: Connectors & Chunking Pipeline
- **`rag/chunking.py`**:
  - Use `tiktoken` (`cl100k_base`) with a safe budget (target 400-600 tokens, 100 token overlap) with headroom for Ollama context windows.
  - **Code chunking**: Python AST-aware chunker for classes and functions; structural delimiter chunking for TypeScript, C#, Go, Java, keeping signatures and file path headers.
  - **Markdown section chunker**: Splits Confluence/Docs by markdown header levels (`#`, `##`, `###`), maintaining breadcrumbs (e.g. `Doc Title > Section > Subsection`) in each chunk.
  - Skip binary files and sensitive secret files (`.env`, `.pem`, `.key`).
- **`ingestion/github_loader.py`**:
  - Uses `ghapi` (`GhApi`) for repository metadata, issues, PRs, and READMEs.
  - For full source code: Performs a shallow `git clone --depth 1` into `data/repos/` and applies the AST/code chunker, bypassing API rate limits and base64 overhead.
- **`ingestion/ado_loader.py`**:
  - `httpx.AsyncClient` client for ADO REST API.
  - Fetches Work Items (Epics, Features, Stories, Bugs, Tasks), iteration paths (Sprints), acceptance criteria, and discussion comments.
  - Clones or fetches ADO Git repositories.
- **`ingestion/confluence_loader.py`**:
  - `httpx.AsyncClient` client for Confluence Cloud/Server REST API.
  - Fetches pages by ID or space key, uses `toolslm` (with HTML2Text fallback) to produce clean markdown, preserving header hierarchy.
- **Concurrency & Dedup**:
  - Use `fastcore.parallel` for efficient local crawling and file reading.
  - SHA256 `content_hash` computation on every chunk before embedding.

### Phase 3: DuckDB Embedded Store & Hybrid Retrieval
- **`rag/duckdb_store.py`**:
  - Table: `documents(id VARCHAR PRIMARY KEY, content TEXT, embedding FLOAT[768], source_url VARCHAR, file_path VARCHAR, doc_type VARCHAR, sprint_id VARCHAR, work_item_id VARCHAR, author VARCHAR, commit_hash VARCHAR, content_hash VARCHAR, created_at TIMESTAMP)`.
  - Dedup check: `SELECT 1 FROM documents WHERE content_hash = ?`. Skip embedding and insertion if unchanged.
  - Dense retrieval: `array_cosine_similarity(embedding, $query_embedding::FLOAT[768])`.
  - Sparse retrieval: DuckDB FTS index (`PRAGMA create_fts_index(...)`) or BM25 keyword query.
- **`rag/ollama_client.py`**:
  - Async client calling Ollama's `/api/embeddings` (batched) and `/api/chat` (streaming).
  - Explicitly sends `keep_alive: "5m"` in payloads so idle models are promptly evicted from RAM.
- **`rag/hybrid_search.py` & `rag/reranker.py`**:
  - Reciprocal Rank Fusion (RRF): Combines top-$N$ dense and top-$N$ sparse matches with $k=60$.
  - Intent routing / SQL metadata filtering (e.g. query explicitly targeting "code", "bugs", "sprint 42", or "confluence").
  - FlashRank local ONNX re-ranking on CPU (top 25 hybrid fused down to top 5 reranked chunks).

### Phase 4: Local RAG Chain & Streaming
- **`rag/rag_pipeline.py`**:
  - System prompt enforcing absolute groundedness: *"Answer ONLY from the provided context; if unsure, state what is missing. You MUST provide inline citations for every claim using `[Source: <path_or_id>]`."*
  - Context assembler prefixing every passage with its exact citation tag:
    - Code: `[Source: <file_path>:<line_range>]`
    - ADO: `[Source: ADO <doc_type> #<work_item_id> - <title>]`
    - Confluence: `[Source: Confluence - <title> (<source_url>)]`
  - Async token generator yielding stream chunks directly from Ollama.

### Phase 5: FastHTML + MonsterUI Frontend & Ingestion UI
- **Single FastHTML App (`app/main.py`)**:
  - Unified Starlette ASGI process running FastHTML + MonsterUI.
  - **Tab 1 — Ingest Sources**:
    - URL input form (GitHub repo URL, ADO Board URL / query ID, Confluence Space/Page URL).
    - Source type selector, credentials status indicator, and trigger button.
    - SSE live progress bar, stage indicator ("Cloning...", "Chunking...", "Embedding..."), document counter, and error logs.
    - Indexed collection statistics card (totals by doc_type, last updated).
  - **Tab 2 — Ask Chatbot**:
    - Clean conversational chat interface.
    - Token-by-token streaming response via SSE.
    - Interactive MonsterUI collapsible citation drawer displaying retrieved code blocks, work items, and Confluence sections with relevance scores and direct source links.
    - Intent and metadata filter toggles (All, Code only, Work items only, Confluence only).
- **CLI (`cli.py`)**:
  - Headless commands: `python cli.py ingest --type github --url <repo_url>`, `python cli.py query "how does authentication work?"`, `python cli.py stats`.

---

## Verification Plan

### Automated Tests
1. **Chunking & Tokenizer**:
   `python -m unittest tests/test_chunking.py`
   - Verifies AST chunking preserves method context and function headers.
   - Verifies markdown section chunking preserves parent headers.
   - Verifies token count limits stay within safe bounds.
2. **DuckDB Store & Deduplication**:
   `python -m unittest tests/test_duckdb_store.py`
   - Verifies table initialization, vector storage, cosine similarity calculations.
   - Verifies SHA256 content deduplication skips existing documents.
3. **Hybrid Search & FlashRank**:
   `python -m unittest tests/test_hybrid_search.py`
   - Verifies dense + sparse RRF fusion ranking.
   - Verifies FlashRank CPU re-ranking executes locally without network calls.
4. **End-to-End Pipeline & Mock Ollama Integration**:
   `python -m unittest tests/test_ingestion.py`
   - Verifies loader error handling (invalid URLs, unreadable binaries, mock ADO/GitHub responses).

### Manual Verification
1. **Ingest Test**: Ingest a local or remote git repository using the UI and CLI. Verify live SSE progress bar and document count updates.
2. **Citation & Groundedness Test**: Submit queries targeting code, work items, and documentation. Confirm inline citations format correctly and open the collapsible citation drawer.
3. **RAM Profiling**: Measure memory usage during ingestion and during chat generation via PowerShell `Get-Process python` to confirm adherence to the 16 GB hardware budget.
