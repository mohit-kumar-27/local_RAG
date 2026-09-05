# Walkthrough: Persistent Chat History Management for Local RAG Chatbot

We have designed, implemented, verified, and synchronized a complete **persistent chat history management** system for the FastHTML + MonsterUI local RAG application. All changes strictly adhere to the existing zero-telemetry, single-process ASGI stack (FastHTML, MonsterUI, DuckDB, httpx, Ollama) and are optimized for a **Windows 11 machine with 16 GB RAM**.

---

## 1. Architecture & Design Overview

The persistent chat history enhancement introduces end-to-end multi-session state, conversation resumption, sliding-window RAG context, inline editing, and deletion to the existing two-tab interface.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                FastHTML ASGI Web App                                    │
│                                                                                         │
│  ┌──────────────────────┐  ┌─────────────────────────────────────────────────────────┐  │
│  │   ChatSidebar        │  │                     ChatMainArea                        │  │
│  │                      │  │                                                         │  │
│  │  [+ New Chat]        │  │  [Scope / Sprint Filters Bar]                           │  │
│  │                      │  │  ─────────────────────────────────────────────────────  │  │
│  │  • Active Chat Title │  │  User Bubble: "How does token chunking work?"           │  │
│  │    5m ago  [Trash]   │  │    [Edit ✏️] [Delete 🗑️]                                 │  │
│  │                      │  │  Assistant Bubble: "Token chunking splits text..."      │  │
│  │  • Older Chat Title  │  │    ▼ Retrieved Context Sources (2 chunks)               │  │
│  │    2h ago  [Trash]   │  │    [Delete 🗑️]                                           │  │
│  │                      │  │  ─────────────────────────────────────────────────────  │  │
│  │  • Yesterday Chat    │  │  [ Prompt Textarea...                        ] [Send ↵] │  │
│  │    1d ago  [Trash]   │  │  (Hidden input: chat_id)                                │  │
│  └──────────────────────┘  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
    ┌──────────────────────┐                                  ┌──────────────────────┐
    │     rag_pipeline     │                                  │     duckdb_store     │
    │                      │                                  │                      │
    │  Sliding Window (10) │ ◄─────── Fetch Prior Turns ───── │  Table: chats        │
    │  Prior Context Q&A   │                                  │  Table: chat_messages│
    │  Grounded Citations  │ ─────── Store Q, A & Citations ─►│  Thread-Safe RLock   │
    │  Ollama Streaming    │                                  │                      │
    └──────────────────────┘                                  └──────────────────────┘
```

---

## 2. DuckDB Schema & Storage Layer

### Schema Specification
Persistent chat tables are initialized idempotently in `rag/duckdb_store.py` (`_init_db`):

```sql
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations JSON,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages(chat_id);
```

> [!IMPORTANT]
> **DuckDB Foreign Key Architecture Decision**:
> In DuckDB, explicit DDL `FOREIGN KEY` constraints do not support `ON DELETE CASCADE`. Furthermore, declaring `FOREIGN KEY (chat_id) REFERENCES chats(id)` causes DuckDB to treat row updates to `chats` (such as `updated_at` or `title`) as a delete + re-insert, resulting in:
> `Constraint Error: Violates foreign key constraint because key "chat_id: ..." is still referenced by a foreign key in a different table`.
> 
> **Solution**: Referential integrity and cascading deletion are managed transactionally in Python (`store.delete_chat` deletes messages before deleting the session inside a transaction). This enables smooth `UPDATE chats SET updated_at = ...` operations while preserving absolute data integrity.

### Thread Safety
- Added `self._lock = threading.RLock()` to `DuckDBStore`.
- All writes (`insert_documents`, `delete_documents_*`, `clear_all`, and all chat CRUD operations) are guarded by `with self._lock:` to ensure concurrent safety under async and background worker operations.

### Data Classes & CRUD Methods
- `ChatSession(id, title, created_at, updated_at)`
- `ChatMessageRecord(id, chat_id, role, content, citations, created_at)`
- Implemented: `create_chat`, `get_chat`, `list_chats`, `update_chat_title`, `touch_chat`, `delete_chat`, `add_chat_message`, `get_chat_messages(limit=...)`, `get_message`, `update_message_content`, `delete_message`, `delete_message_pair`, `delete_messages_after`.

---

## 3. Sliding-Window Context & 16 GB RAM Budget

### Memory Impact Analysis
| Component | Tokens / Size | RAM Impact | Rationale |
| :--- | :--- | :--- | :--- |
| **Window Size (`CHAT_HISTORY_WINDOW_SIZE=10`)** | 5 User + 5 Assistant turns | ~1,500 - 2,500 tokens | Balanced conversational continuity without runaway prompt expansion |
| **Ollama Context Window** | 4,096 - 8,192 tokens | ~500 MB extra KV cache | Plenty of room for top-4 retrieved RAG chunks (~1,500 tokens) + system instructions |
| **16 GB System Safety Headroom** | — | **4.0 - 5.5 GB free** | Safe with `llama3.1:8b` Q4 (~5.2 GB resident) and DuckDB (~250 MB) |

### Context Assembly in `rag_pipeline.py`
When `chat_id` is supplied to `answer_stream`:
1. The pipeline queries DuckDB for the last `CHAT_HISTORY_WINDOW_SIZE` messages for that chat session (excluding the current user prompt).
2. Turns are formatted as structured `{"role": "user"|"assistant", "content": "..."}` objects.
3. System prompt with retrieved RAG context is injected as the grounding prompt for the latest turn:
   ```python
   messages = [
       {"role": "system", "content": "You are a confidential code and project intelligence assistant..."},
       *history_messages,  # Prior 10 turns
       {"role": "user", "content": user_prompt_with_retrieved_context}
   ]
   ```

---

## 4. UI Components & Interaction Workflows

### 4.1 Chat Sidebar (`ChatSidebar`)
- Positioned on the left side of the "Ask Chatbot" tab (`w-72 md:w-80 shrink-0 border-r`).
- **Collapsible Toggle**:
  - **Collapse Button (`[ ◀ ]`)**: Embedded directly in the sidebar header next to "Sessions", collapsing the sidebar smoothly (`width: 0`, `opacity: 0`, `overflow: hidden`).
  - **Expand Button (`[ 💬 Sessions ▶ ]`)**: Automatically revealed at the top-left of the conversation filter bar when the sidebar is collapsed, allowing one-click restoration.
  - **Keyboard Shortcut**: Press `Ctrl+B` (or `Cmd+B`) anywhere on the chat page to instantly toggle the sidebar.
  - **Persistent State**: The collapsed/expanded state is saved to `localStorage` (`chat_sidebar_collapsed`) and restored automatically across page reloads and HTMX out-of-band swaps.
  - **Responsive Behavior**: On mobile viewports (`< 768px`), collapsing hides the sidebar completely (`display: none`), maximizing vertical screen space for the message timeline.
- **"+ New Chat" Button**: Clears the active chat state and loads a fresh conversation window.
- **Session List**: Rendered in reverse chronological order (`updated_at DESC`).
- **Relative Timestamps**: Formatted dynamically via `format_relative_time` ("Just now", "5m ago", "2h ago", "3d ago", "Aug 12").
- **Title Display**: Truncated cleanly to ~60 characters with trailing ellipsis.
- **Active Highlighting**: Distinct active pill styling (`bg-primary/10 border-primary text-primary font-medium`).
- **Session Deletion**: Trash can button triggering `hx-delete="/api/chats/{id}"` with an interactive native confirmation prompt (`hx-confirm`).

### 4.2 Collapsible Citation Drawer Replay (`AssistantMessageBubble`)
- Citations are serialized to JSON on creation (`file_path`, `source_type`, `score`, `content`).
- When historical sessions are resumed, citations are deserialized and rendered into the native collapsible `Details`/`Summary` accordion drawer with high-contrast badges (`CODE`, `TICKET`, `CONFLUENCE`) and dark IDE code snippets.

### 4.4 Inline User Message Edit Workflow (`EditMessageForm`)
1. User clicks the **Edit (✏️)** button on any of their previous prompts.
2. FastHTML swaps the message bubble with an inline `textarea` and "Save & Re-generate" / "Cancel" buttons (`GET /api/chats/{chat_id}/messages/{message_id}/edit-form`).
3. On submission (`POST /api/chats/{chat_id}/messages/{message_id}/edit`):
   - User message content is updated in DuckDB.
   - All subsequent assistant and user messages in the session are deleted (`delete_messages_after`).
   - HTMX swaps out the edited message and subsequent turns, and immediately streams a fresh RAG answer via SSE.

### 4.5 Inline AI Response Edit Workflow (`EditAssistantMessageForm`)
1. User hovers or focuses on any AI response bubble and clicks the **Edit (✏️)** button in the top-right action toolbar.
2. FastHTML replaces the assistant bubble with an inline markdown editor textarea (`rows=8`, font-mono) along with "Cancel" and "Save Changes" controls (`GET /api/chats/{chat_id}/assistant-messages/{message_id}/edit-form`).
3. On submission (`POST /api/chats/{chat_id}/assistant-messages/{message_id}/edit`):
   - Assistant message content is updated directly in DuckDB (`store.update_message_content`).
   - The session timestamp is bumped (`store.touch_chat`).
   - The original citations drawer and all subsequent conversation turns are **preserved** (unlike user prompt edits which prune subsequent turns).
   - FastHTML re-renders the markdown and citation drawer into the timeline with zero page reload.
4. If "Cancel" is clicked (`GET /api/chats/{chat_id}/assistant-messages/{message_id}/cancel-edit`), the original unedited bubble is restored instantly.
5. **Sliding-Window Memory Benefit**: Any edits made to an AI response immediately feed into the sliding-window memory (`rag_pipeline.answer_stream`), allowing users to correct or steer facts for subsequent questions in the conversation.

### 4.6 Message Deletion
- **User Prompt**: Clicking Delete prompts confirmation ("Delete this message and its response?"). On approval, deletes both the user message and its paired assistant reply (`delete_message_pair`).
- **Assistant Reply**: Clicking Delete prompts confirmation ("Delete this response?"). On approval, deletes only the assistant response (`delete_message`).

### 4.7 Fixed Action Button Alignment & Positioning
Previously, edit and delete buttons could appear at unexpected positions away from the text:
1. **User Message Hover Scope**:
   - *Problem*: The `group` hover trigger was declared on the full-width row (`w-full`). Hovering over empty whitespace on the left side of the screen triggered `group-hover:opacity-100`, causing buttons to pop up far to the right. Furthermore, `self-center` floated buttons in the vertical middle of tall messages.
   - *Fix*: Scoped `group` strictly to an inner wrapper around the message bubble (`max-w-2xl`), eliminating empty-space hover triggers. Replaced `self-center` with `mt-1.5 flex-shrink-0` so action buttons are permanently docked right at the top-left of the prompt.
2. **AI Response Header Toolbar**:
   - *Problem*: Action buttons relied on `absolute top-3 right-3`. When `relative` was omitted from `#response-box` during `post_edit_message`, the buttons escaped to the top-right corner of the entire browser window (`top: 12px; right: 12px`). Additionally, `absolute` could overlap the first line of markdown headings.
   - *Fix*: Replaced floating absolute positioning with an **in-flow flexbox header bar** (`flex items-center justify-between pb-2 border-b`) directly above the markdown text. Left: `🤖 AI Response` indicator (with pulsing `● Generating...` during streaming); Right: `[✏️ Edit] [🗑️]` hover action buttons. Because it is in normal flexbox flow, it physically cannot escape, overlap text, or detach from the response card.

### 4.8 High-Contrast Visible Scrollbars on Both Pages
- **Tab 1 (Ingest Sources)**:
  - Configured `#tab-content` to use `overflow-y-scroll min-h-0 scroll-smooth scroll-page` with `scrollbar-gutter: stable !important`.
  - Guarantees a persistent, high-contrast, modern vertical scrollbar on the right side of the screen across the entire page (stats card, source forms, live progress).
- **Tab 2 (Ask Chatbot)**:
  - Configured `#chat-main-area` to use `overflow-y-scroll scroll-smooth` with `scrollbar-gutter: stable !important`.
  - Guarantees a persistent, modern vertical scrollbar on the conversation timeline while keeping the scope/sprint filter and prompt textarea fixed in sticky docking.
  - Configured `#chat-sidebar-list` to use `overflow-y-auto min-h-0` with custom scrollbar styling for long session lists.
- **Fixed Tab-Swapping Scroll State**:
  - In `TabNavigation`, added `hx_swap="outerHTML"` to both tab navigation links.
  - In `get_tab`, wrapped tab contents in `<div id="tab-content" class="...">` with `overflow-y-scroll` for Ingest and `overflow-hidden` for Chat, ensuring switching tabs via HTMX swaps the container's scroll classes cleanly without freezing the view.
- **Custom Modern Cross-Browser Scrollbar CSS**:
  - Custom WebKit/Chromium (`::-webkit-scrollbar`, `10px` width, `#f1f5f9` track, `#94a3b8` thumb with 2px borders, dark mode variants).
  - Modern Firefox standards (`scrollbar-width: thin; scrollbar-color: #94a3b8 #f1f5f9;`).
  - Disabled Windows 11 Chromium auto-hide overlay behavior by explicitly declaring custom scrollbar pseudo-elements on all scrollable containers.

---

## 5. Verification & Test Results

### 5.1 Automated Unit & Integration Tests
A dedicated test suite in `tests/test_chat_history.py` verifies:
1. `test_chat_creation_and_retrieval`: Chat lifecycle, title truncation, timestamps.
2. `test_add_and_retrieve_messages_with_citations`: JSON serialization/deserialization, citation structure preservation.
3. `test_sliding_window_limit`: Verifies message fetching strictly honors `limit=10`.
4. `test_update_message_content`: Verifies prompt editing.
5. `test_delete_message_pair`: Verifies atomic deletion of user message and paired assistant reply.
6. `test_delete_messages_after`: Verifies branching/pruning logic when editing an earlier user message.
7. `test_delete_chat_cascade`: Verifies complete cleanup of all messages when a chat is deleted.
8. `test_chat_sidebar_kwargs_and_oob`: Verifies `ChatSidebar` accepts `**kwargs` and renders `hx-swap-oob="true"`.
9. `test_post_chat_and_session_routes`: Verifies `POST /api/chat`, `GET /api/chats/new`, and `GET /api/chats/{id}` via `TestClient`.
10. `test_scrollbars_and_tab_swapping`: Verifies persistent `overflow-y-scroll` and CSS scrollbars across both Ingest and Chat pages and during HTMX tab transitions.
11. `test_assistant_message_editing`: Verifies edit form retrieval, cancelation, post-edit DuckDB update, citation preservation, and UI re-rendering.
12. `test_collapsible_sidebar`: Verifies collapse and expand buttons, CSS transition rules, JS state management, and localStorage persistence.

### 5.2 Full Test Suite Execution
Executed in `C:\Users\krmoh\.gemini\antigravity\scratch\local_rag_system`:

```powershell
python -m unittest discover tests -v
# Result: Ran 32 tests in 31.354s -> OK (32/32 Passed)
```

---

## 6. Workspace Isolation

Per user instruction, automatic synchronization to the external workspace (`C:\Users\krmoh\OneDrive\Documents\my_projects\local_RAG`) has been stopped. All active code modifications, scrollbar enhancements, bug fixes, and test suites are developed and tested within `C:\Users\krmoh\.gemini\antigravity\scratch\local_rag_system`.

---

## 7. How to Use Persistent Chat History

1. **Start the Application**:
   ```powershell
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
2. **Start a Conversation**:
   - Navigate to `http://localhost:8000`.
   - Ask a question. A chat session is automatically created, titled with the first ~60 characters of your prompt, and displayed in the left sidebar with a relative timestamp ("Just now").
3. **Switch Between Sessions**:
   - Click any session in the sidebar to load its full message timeline and citation drawers.
   - Click **+ New Chat** to clear the timeline and start a clean conversation.
4. **Edit Earlier Prompts**:
   - Hover over any user message and click the ✏️ icon.
   - Modify the question and click **Save & Re-generate**. The timeline updates and a new response streams in real time.
5. **Delete Messages or Sessions**:
   - Click the 🗑️ icon next to a message or in the sidebar next to a chat title to delete it with confirmation.
