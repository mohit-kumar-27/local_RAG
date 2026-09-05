# User Guide: Local Confidential RAG System & Chatbot UI

This guide provides a comprehensive walkthrough of the user interface for the **Local Confidential RAG System & Chatbot**, detailing all features, settings, workflow options, and keyboard shortcuts across both primary views: **Ingest Sources** and **Ask Chatbot**.

---

## Table of Contents
1. [Global Header & Navigation Bar](#1-global-header--navigation-bar)
2. [Page 1: Ingest Sources Tab](#2-page-1-ingest-sources-tab)
   - [Knowledge Base Statistics Card](#knowledge-base-statistics-card)
   - [Ingest New Source Form](#ingest-new-source-form)
   - [Source Types & Parameter Guide](#source-types--parameter-guide)
   - [Real-Time SSE Progress & Log Viewer](#real-time-sse-progress--log-viewer)
3. [Page 2: Ask Chatbot Tab](#3-page-2-ask-chatbot-tab)
   - [Target Scope & Sprint Filters](#target-scope--sprint-filters)
   - [Chat Timeline & Streaming Responses](#chat-timeline--streaming-responses)
   - [Retrieved Context Sources (Citation Drawer)](#retrieved-context-sources-citation-drawer)
   - [Prompt Input Box & Keyboard Shortcuts](#prompt-input-box--keyboard-shortcuts)
   - [Smart Auto-Scrolling Behavior](#smart-auto-scrolling-behavior)
4. [Hardware & Performance Tuning](#4-hardware--performance-tuning)
5. [End-to-End Walkthrough Example](#5-end-to-end-walkthrough-example)

---

## 1. Global Header & Navigation Bar

The top header is permanently visible at the top of the application across both views.

| Element | Description & Functionality |
| :--- | :--- |
| **App Title & Branding** | Displays `Local Confidential RAG & Chatbot` and supported integration logos. |
| **Confidential Badge** | `Confidential · Zero Telemetry` confirms all outbound cloud AI calls and telemetry are disabled. |
| **Ollama Status Badge** | - Green (`Ollama Active`): Ollama is reachable at `http://localhost:11434`.<br>- Red (`Ollama Disconnected`): Ollama is offline or unreachable. |
| **Active Model Badge** | Displays currently active generation model (e.g. `Model: llama3.1:8b` or `llama3.2:3b`). |
| **RAM Mode Toggle Button** | **`RAM: 8B Standard (Toggle)`** / **`RAM: 3B Low-RAM (Toggle)`**:<br>Clicking this switches between `llama3.1:8b` (~5.5 GB RAM) and `llama3.2:3b` (~2.2 GB RAM) on the fly without page reload or losing chat history. |
| **Segmented Pill Navigation** | Switch between `💬 Ask Chatbot` and `📥 Ingest Sources`. Active page features a high-contrast pill, pulsing dot, and instant browser title update. |

---

## 2. Page 1: Ingest Sources Tab

Access via the **Ingest Sources** button in the navigation bar or by visiting `http://localhost:8000/?tab=ingest`.

### Knowledge Base Statistics Card
Provides an instant overview of your local DuckDB vector database:
- **Total Chunks**: Total number of indexed vector embeddings.
- **Code Chunks**: Source code functions, classes, and structural blocks.
- **ADO Work Items**: Indexed Azure DevOps Epics, Stories, Bugs, Tasks, and Sprints.
- **Confluence Wiki**: Confluence page sections and documentation chunks.
- **DuckDB Storage**: Exact file size of the local database on disk (e.g., `1.42 MB`).
- **Refresh Button**: Fetches fresh metrics from DuckDB without refreshing the browser.
- **Clear DB Button**: Prompts for confirmation and purges all indexed vectors and metadata, resetting DuckDB to an empty state.

### Ingest New Source Form
Configure and initiate ingestion pipelines:

| Form Field | Type | Description |
| :--- | :--- | :--- |
| **Source Type** | Dropdown | Select connector: GitHub Repository, Azure DevOps Board, Azure DevOps Repo, or Confluence Space. |
| **Source URL / Key** | Text (Required) | Full URL to repository/board or Confluence space key. |
| **Branch / Filter** | Text (Optional) | Git branch name (`main`), sprint name (`Sprint 42`), or CQL filter. |
| **PAT Override** | Password (Optional) | Personal Access Token override for private sources. If omitted, credentials from `.env` are used. |
| **Start Ingestion** | Button | Submits job to the background worker and opens real-time progress stream. |

### Source Types & Parameter Guide

#### 1. GitHub Repository (`github`)
- **URL Format**: `https://github.com/owner/repo`
- **Branch / Filter**: (Optional) e.g., `main`, `master`, or `release/v2.0` (defaults to default branch).
- **Mechanism**: Executes a fast shallow git clone (`--depth 1`) directly into local cache. Parses Python files using AST chunking to preserve functions/classes and other languages structurally.
- **Incremental Sync**: When re-ingesting an existing repo, it performs `git pull` and uses `git diff --name-status` to identify only added, modified, or deleted files. Unchanged files are skipped; modified files have stale chunks pruned; deleted files are purged.

#### 2. Azure DevOps Board (`ado_board`)
- **URL Format**: `https://dev.azure.com/org/project` or board URL `https://dev.azure.com/org/project/_boards/...`
- **Branch / Filter**: (Optional) Specific Sprint name (e.g., `Sprint 42`) to restrict indexing to that sprint's items.
- **Mechanism**: Queries Azure DevOps REST API via `httpx.AsyncClient`. Ingests Work Items, Bugs, User Stories, acceptance criteria, state transitions, and discussion comments.

#### 3. Azure DevOps Git Repo (`ado_repo`)
- **URL Format**: `https://dev.azure.com/org/project/_git/repo_name`
- **Branch / Filter**: (Optional) Git branch (defaults to `main`).
- **Mechanism**: Shallow clones the ADO Git repository using authenticated HTTP with your ADO PAT token.

#### 4. Atlassian Confluence (`confluence`)
- **URL Format / Key**: Confluence Space Key (e.g., `ENG`, `DOCS`, `ARCH`) or base URL `https://your-org.atlassian.net/wiki`.
- **Branch / Filter**: (Optional) CQL query filter or specific Page Title prefix.
- **Mechanism**: Fetches wiki pages, converts HTML to Markdown with `toolslm.download.html2md`, and splits sections preserving heading hierarchy breadcrumbs (`# Title > ## Architecture > ### Auth`).

### Real-Time SSE Progress & Log Viewer
Once **Start Ingestion** is clicked:
1. **Live Progress Bar**: Visual progress percentage from `0%` to `100%`.
2. **Lifecycle Status Badge**:
   - `QUEUED`: Background worker is initializing.
   - `RUNNING`: Actively cloning, parsing, hashing, or embedding.
   - `COMPLETED`: Ingestion finished successfully.
   - `FAILED`: Error occurred (displays error alert with exact cause).
3. **Current Stage Indicator**: Real-time phase description (e.g., *Cloning repository...*, *Generating embeddings via Ollama (batch 12/40)...*, *Committing chunks to DuckDB...*).
4. **Log Terminal**: Scrolling output terminal displaying file-by-file status, SHA-256 deduplication cache hits, and pruned stale vectors.
5. **Update Stats Button**: Automatically appears upon completion to update the statistics card in one click.

---

## 3. Page 2: Ask Chatbot Tab

Access via the **Ask Chatbot** button in the navigation bar or by visiting `http://localhost:8000/?tab=chat`.

### Target Scope & Sprint Filters
Located in the sticky top filter bar:
- **Target Scope (Dropdown)**:
  - **All Sources (Auto-route)** *(Default)*: Uses hybrid search (Dense Vector + BM25 FTS + RRF) across all indexed code, tickets, and wiki pages.
  - **Source Code Only**: Restricts retrieval strictly to code chunks (`doc_type = 'code'`).
  - **ADO Work Items / Bugs**: Restricts retrieval strictly to tickets, bugs, and user stories (`doc_type = 'ticket'`).
  - **Confluence Wiki Pages**: Restricts retrieval strictly to documentation (`doc_type = 'confluence'`).
- **Sprint Filter (Text Input)**:
  - Enter a sprint name (e.g., `Sprint 42`) to constrain queries to tickets or items assigned to that sprint.

### Chat Timeline & Streaming Responses
- **Welcome Guidance Card**: Explains system capabilities and provides clickable example questions.
- **User Prompt Message**: Formatted speech bubble with compact padding, distinct background, and `YOU` badge.
- **Assistant Streaming Output**:
  - Responses stream token-by-token using Starlette Server-Sent Events (SSE).
  - Grounded strictly in indexed sources—no hallucinated internal facts.
  - Formatted in GitHub Flavored Markdown (code snippets with syntax highlighting, bullet points, headers, tables).
  - Inline source citations (e.g., `[Source: auth.py:45-78]` or `[Source: BUG-1042]`).

### Retrieved Context Sources (Citation Drawer)
Directly beneath every assistant answer is a collapsible **Retrieved Context Sources** accordion:
1. **Header Info**: Displays source type badge (`CODE`, `TICKET`, `CONFLUENCE`), citation tag, and FlashRank relevance score (e.g., `Relevance: 0.895`).
2. **Click to Expand**:
   - **Source URL**: Clickable link to GitHub/ADO/Confluence.
   - **File Path / Ticket ID / Sprint**: Exact file path and line numbers, or work item ID and sprint.
   - **Verbatim Context Chunk**: Exact code block or markdown section passed to the LLM.

### Prompt Input Box & Keyboard Shortcuts
Located in the sticky bottom card:
- **Auto-Expanding Input**: Starts at a compact `44px` height and smoothly expands up to `160px` as multi-line prompts are typed.
- **Keyboard Shortcuts**:
  - `Enter ↵`: Submits prompt immediately.
  - `Shift + Enter`: Inserts a newline for multi-line questions or code snippets.
- **Submit Button ("Ask →")**: One-click submission alternative.
- **Automatic Form Reset**: Clears the input and restores single-line height upon submission.

### Smart Auto-Scrolling Behavior
- **Universal Page Scrollbar**: Custom styled, visible scrollbar on the right edge of the window.
- **Follow Streaming Tokens**: The page automatically scrolls down smoothly as tokens stream in.
- **User Scroll Preservation**: If you intentionally scroll up to inspect earlier messages or open a citation drawer, auto-scrolling yields, allowing you to read uninterrupted without being pulled down.

---

## 4. Hardware & Performance Tuning

Designed specifically for **Windows 11 with 16 GB RAM**:

| Scenario | Recommendation |
| :--- | :--- |
| **Normal Use (10+ GB Free RAM)** | Use default **8B Standard** mode (`llama3.1:8b`). Delivers superior reasoning and code analysis. |
| **Constrained RAM (< 4 GB Free RAM)** | Click the **RAM: (Toggle)** button in the header bar to activate **3B Low-RAM** mode (`llama3.2:3b`). Peak RAM drops to ~2.2 GB. |
| **Memory Headroom Protection** | Ollama is configured with `keep_alive="5m"`. If idle for 5 minutes, models automatically unload from RAM. |
| **Database Boundaries** | Embedded DuckDB is bounded by `PRAGMA max_memory='2GB'`. |

---

## 5. End-to-End Walkthrough Example

### Ingesting a Codebase
1. Open the **Ingest Sources** tab.
2. Select `GitHub Repository (ghapi + git clone)`.
3. Enter `https://github.com/fastai/fastcore` into **Source URL**.
4. Leave branch blank (defaults to `main`).
5. Click **Start Ingestion**.
6. Observe the live progress bar and logs as repository files are cloned, chunked with AST parsing, hashed with SHA-256, and embedded.
7. Once finished, click **Update Stats** to see the new chunk count.

### Asking a Technical Question
1. Click the **Ask Chatbot** tab.
2. Under **Target Scope**, select `Source Code Only` (or leave as `All Sources`).
3. In the input box, type:
   ```text
   How does the @patch decorator work in fastcore?
   ```
4. Press `Enter ↵`.
5. Watch the explanation stream in token-by-token with code examples and inline citations.
6. Click the **📚 Retrieved Context Sources** accordion below the response to verify the exact source file (`fastcore/basics.py`), line numbers, and relevance score.
