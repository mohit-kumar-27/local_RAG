"""
FastHTML single ASGI application.
Combines Starlette backend routes, MonsterUI component rendering,
and real-time SSE token/progress streaming into a single process.
"""

import asyncio
import html
import os
import re
import urllib.parse
import uuid
from typing import Optional

from fasthtml.common import (
    A, Body, Button, Div, FastHTML, Form, H1, H2, H3, Html, Input, NotStr, Option, P,
    Script, Select, Span, Style, Textarea, Title, fast_app, sse_message
)
import mistletoe
import monsterui.all as ui
from starlette.requests import Request
from starlette.responses import StreamingResponse

import config
from app.background import create_job, execute_ingestion, get_job
from app.ui_components import (
    AppHeader, AssistantMessageBubble, ChatMainArea, ChatSidebar, ChatTab, CitationDrawer, CollectionStatsCard,
    EditAssistantMessageForm, EditMessageForm, IngestProgressSSEComponent, IngestProgressUpdateCard, IngestionTab,
    TabNavigation, UserMessageBubble, deserialize_citations, format_inline_citations, serialize_citations
)
from ingestion.ado_loader import AdoBoardLoader, AdoRepoLoader
from ingestion.confluence_loader import ConfluenceLoader
from ingestion.github_loader import GithubCodeLoader
from rag.duckdb_store import DuckDBStore
from rag.hybrid_search import HybridSearcher
from rag.ollama_client import OllamaClient
from rag.rag_pipeline import RAGPipeline
from rag.reranker import LocalReranker

# Initialize Core Services (Single embedded process)
store = DuckDBStore()
ollama = OllamaClient()
reranker = LocalReranker()
hybrid_searcher = HybridSearcher(store=store, ollama_client=ollama, reranker=reranker)
rag_pipeline = RAGPipeline(hybrid_searcher=hybrid_searcher, ollama_client=ollama)

# Single FastHTML ASGI App
app, rt = fast_app(
    hdrs=[
        *ui.Theme.blue.headers(),
        # Custom CSS for full-page layout, smooth scrolling, and prominent modern scrollbars
        Style("""
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
        }

        /* Universal high-contrast visible scrollbars across all browsers and both pages */
        * {
            scrollbar-width: thin;
            scrollbar-color: #94a3b8 #f1f5f9;
        }
        .dark *, [data-theme="dark"] * {
            scrollbar-color: #64748b #1e293b;
        }

        /* WebKit / Chromium (Chrome, Edge, Brave, Opera, Safari) */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 6px;
        }
        .dark ::-webkit-scrollbar-track, [data-theme="dark"] ::-webkit-scrollbar-track {
            background: #1e293b;
            border-radius: 6px;
        }
        ::-webkit-scrollbar-thumb {
            background: #94a3b8;
            border-radius: 6px;
            border: 2px solid #f1f5f9;
        }
        .dark ::-webkit-scrollbar-thumb, [data-theme="dark"] ::-webkit-scrollbar-thumb {
            background: #64748b;
            border: 2px solid #1e293b;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #64748b;
        }
        .dark ::-webkit-scrollbar-thumb:hover, [data-theme="dark"] ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }

        /* Dedicated container scrollbars and stable gutter on both pages */
        #tab-content.scroll-page {
            overflow-y: scroll !important;
            scrollbar-gutter: stable;
        }
        #chat-main-area {
            overflow-y: scroll !important;
            scrollbar-gutter: stable;
        }
        #chat-sidebar-list {
            overflow-y: auto !important;
            scrollbar-gutter: stable;
        }
        """),
        # HTMX SSE Extension
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.2/sse.js"),
        # Auto-scroll & SSE close handling
        Script("""
        document.addEventListener('htmx:afterSwap', function(evt) {
            const chatMainArea = document.getElementById('chat-main-area');
            if (chatMainArea && evt.detail && evt.detail.target) {
                if (evt.detail.target.id === 'chat-history' || evt.detail.target.closest('#chat-history')) {
                    const isNearBottom = chatMainArea.scrollHeight - chatMainArea.scrollTop - chatMainArea.clientHeight < 240;
                    if (isNearBottom) {
                        chatMainArea.scrollTop = chatMainArea.scrollHeight;
                    }
                }
            }
        });
        document.addEventListener('htmx:sseClose', function(evt) {
            if (evt.target) {
                evt.target.removeAttribute('sse-connect');
            }
        });
        """),
    ],
    live=False,
)


def MainLayout(active_tab: str = "chat", active_chat_id: Optional[str] = None, ollama_ok: bool = True):
    """Main application shell with full-viewport layout, persistent sessions, and scrollable content."""
    stats = store.get_collection_stats()
    active = "ingest" if active_tab == "ingest" else "chat"
    if active == "ingest":
        content = IngestionTab(stats)
        page_title = "Ingest Sources | Local Confidential RAG"
        tab_content_cls = "flex-1 overflow-y-scroll min-h-0 scroll-smooth scroll-page"
    else:
        chats = store.list_chats()
        messages = store.get_chat_messages(active_chat_id) if active_chat_id else []
        content = ChatTab(chats=chats, active_chat_id=active_chat_id, messages=messages)
        page_title = "Ask Chatbot | Local Confidential RAG"
        tab_content_cls = "flex-1 overflow-hidden min-h-0"

    return Title(page_title, id="app-page-title"), Script(f'document.title = "{page_title}";'), Div(
        id="main-content",
        cls="h-screen bg-base-200 text-base-content flex flex-col font-sans overflow-hidden",
    )(
        AppHeader(ollama_connected=ollama_ok, current_model=config.get_active_llm_model(), active_tab=active),
        TabNavigation(active_tab=active),
        Div(
            id="tab-content",
            cls=tab_content_cls,
        )(
            content
        ),
    )


@rt("/")
async def get(tab: Optional[str] = "chat", chat_id: Optional[str] = None):
    """Main route serving Ingest and Chatbot tabs with optional active chat session."""
    ok, _, _ = await ollama.check_connection()
    return MainLayout(active_tab=tab or "chat", active_chat_id=chat_id, ollama_ok=ok)


@rt("/tab/{tab_name}")
async def get_tab(tab_name: str, req: Request, chat_id: Optional[str] = None):
    """Swaps tab content, updates navigation active highlight, and sets page title."""
    active = "ingest" if tab_name == "ingest" else "chat"
    page_title = "Ingest Sources | Local Confidential RAG" if active == "ingest" else "Ask Chatbot | Local Confidential RAG"

    # If direct browser navigation to /tab/... without HTMX, return full layout
    if not req.headers.get("HX-Request"):
        ok, _, _ = await ollama.check_connection()
        return MainLayout(active_tab=active, active_chat_id=chat_id, ollama_ok=ok)

    stats = store.get_collection_stats()
    if active == "ingest":
        content = IngestionTab(stats)
        tab_content_cls = "flex-1 overflow-y-scroll min-h-0 scroll-smooth scroll-page"
    else:
        chats = store.list_chats()
        messages = store.get_chat_messages(chat_id) if chat_id else []
        content = ChatTab(chats=chats, active_chat_id=chat_id, messages=messages)
        tab_content_cls = "flex-1 overflow-hidden min-h-0"

    tab_content = Div(
        id="tab-content",
        cls=tab_content_cls,
    )(content)

    return (
        Title(page_title, id="app-page-title"),
        Script(f'document.title = "{page_title}";'),
        TabNavigation(active_tab=active, hx_swap_oob="true"),
        tab_content,
    )


@rt("/api/stats")
def get_stats():
    """Returns updated stats card."""
    stats = store.get_collection_stats()
    return CollectionStatsCard(stats)


@rt("/api/clear")
def post_clear():
    """Clears DuckDB database."""
    store.clear_all()
    stats = store.get_collection_stats()
    return CollectionStatsCard(stats)


@rt("/api/toggle_ram_mode")
async def post_toggle_ram(current_tab: Optional[str] = "chat"):
    """Toggles LOW_RAM_MODE between 8B and 3B models while preserving current active page."""
    config.LOW_RAM_MODE = not config.LOW_RAM_MODE
    ok, _, _ = await ollama.check_connection()
    active = "ingest" if current_tab == "ingest" else "chat"
    return MainLayout(active_tab=active, ollama_ok=ok)


@rt("/api/ingest")
async def post_ingest(
    source_type: str,
    url: str,
    branch_or_filter: Optional[str] = None,
    token_override: Optional[str] = None,
):
    """Starts background ingestion job and returns the SSE connection container."""
    url = url.strip()
    if not url:
        return Div(cls="uk-alert uk-alert-danger text-sm p-3 rounded")("Error: Source URL is required.")

    # Create loader
    try:
        if source_type == "github":
            loader = GithubCodeLoader(
                repo_url_or_slug=url,
                pat_token=token_override or None,
                branch=branch_or_filter or None,
            )
        elif source_type == "ado_board":
            loader = AdoBoardLoader(
                board_url_or_query=url,
                pat_token=token_override or None,
            )
        elif source_type == "ado_repo":
            loader = AdoRepoLoader(
                repo_url=url,
                pat_token=token_override or None,
                branch=branch_or_filter or "main",
            )
        elif source_type == "confluence":
            loader = ConfluenceLoader(
                url_or_space=url,
                api_token=token_override or None,
            )
        else:
            return Div(cls="uk-alert uk-alert-danger text-sm p-3 rounded")(f"Unsupported source type: {source_type}")
    except Exception as e:
        return Div(cls="uk-alert uk-alert-danger text-sm p-3 rounded")(f"Failed to initialize loader: {str(e)}")

    # Register job and launch background task
    job = create_job(source_type=source_type, url=url)
    asyncio.create_task(execute_ingestion(job.id, loader, store, ollama))

    return IngestProgressSSEComponent(job.id)


@rt("/api/ingest/stream/{job_id}")
async def get_ingest_stream(job_id: str):
    """SSE endpoint streaming ingestion job progress to browser."""
    async def event_generator():
        while True:
            job = get_job(job_id)
            if not job:
                yield sse_message(Div("Job not found."))
                break

            card = IngestProgressUpdateCard(
                job_id=job.id,
                stage=job.stage,
                progress=job.progress,
                status=job.status,
                logs=job.logs,
                error_message=job.error_message,
            )
            yield sse_message(card)

            if job.status in ("completed", "failed"):
                yield "event: close\ndata: finished\n\n"
                break
            await asyncio.sleep(0.6)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@rt("/api/chats/new", methods=["GET"])
def get_new_chat():
    """Resets chat area to clean 'New Chat' state and clears active sidebar highlight."""
    chats = store.list_chats()
    return (
        ChatSidebar(chats=chats, active_chat_id=None, hx_swap_oob="true"),
        ChatMainArea(messages=[], active_chat_id=None),
    )


@rt("/api/chats/{chat_id}", methods=["GET"])
def get_chat_session(chat_id: str):
    """Loads past chat session messages and highlights session in sidebar."""
    chats = store.list_chats()
    messages = store.get_chat_messages(chat_id)
    return (
        ChatSidebar(chats=chats, active_chat_id=chat_id, hx_swap_oob="true"),
        ChatMainArea(messages=messages, active_chat_id=chat_id),
    )


@rt("/api/chats/{chat_id}", methods=["DELETE"])
def delete_chat_session(chat_id: str, active_chat_id: Optional[str] = None):
    """Deletes a chat session and cascades to all its messages."""
    store.delete_chat(chat_id)
    remaining_chats = store.list_chats()
    new_active = None if (active_chat_id == chat_id or not active_chat_id) else active_chat_id
    messages = store.get_chat_messages(new_active) if new_active else []
    return ChatTab(chats=remaining_chats, active_chat_id=new_active, messages=messages)


@rt("/api/chats/{chat_id}/messages/{message_id}/edit-form", methods=["GET"])
def get_edit_message_form(chat_id: str, message_id: str):
    """Returns inline edit form replacing the user message bubble."""
    msg = store.get_message(message_id)
    if not msg:
        return Div("Message not found", cls="text-xs text-rose-500")
    return EditMessageForm(chat_id=chat_id, message_id=message_id, current_content=msg.content)


@rt("/api/chats/{chat_id}/messages/{message_id}/cancel-edit", methods=["GET"])
def get_cancel_edit(chat_id: str, message_id: str):
    """Cancels editing and restores original user message bubble."""
    msg = store.get_message(message_id)
    if not msg:
        return Div()
    return UserMessageBubble(msg=msg, chat_id=chat_id)


@rt("/api/chats/{chat_id}/messages/{message_id}/edit", methods=["POST"])
async def post_edit_message(
    chat_id: str,
    message_id: str,
    new_content: str,
    doc_type_filter: Optional[str] = "all",
    sprint_filter: Optional[str] = None,
):
    """
    Edits a user message, prunes subsequent turns from DuckDB,
    and triggers a fresh RAG generation with SSE streaming.
    """
    new_prompt = new_content.strip()
    store.update_message_content(message_id, new_prompt)
    store.delete_messages_after(chat_id, message_id)

    stream_id = str(uuid.uuid4())[:8]
    assistant_msg_id = str(uuid.uuid4())[:8]
    encoded_query = urllib.parse.quote(new_prompt)
    encoded_doc_filter = urllib.parse.quote(doc_type_filter or "all")
    encoded_sprint = urllib.parse.quote(sprint_filter.strip() if sprint_filter else "")

    stream_url = f"/api/chat/stream/{stream_id}?chat_id={chat_id}&assistant_msg_id={assistant_msg_id}&q={encoded_query}&doc_type={encoded_doc_filter}&sprint={encoded_sprint}"
    assistant_placeholder = Div(
        id=f"assistant-bubble-{assistant_msg_id}",
        hx_ext="sse",
        sse_connect=stream_url,
        sse_swap="message",
        sse_close="close",
        hx_target=f"#response-box-{stream_id}",
        cls="group flex items-start space-x-3 w-full my-2",
    )(
        Div(cls="w-8 h-8 rounded-full bg-slate-800 dark:bg-slate-700 text-white flex-shrink-0 flex items-center justify-center font-black font-mono text-[11px] shadow-sm mt-0.5")("AI"),
        Div(
            id=f"response-box-{stream_id}",
            cls="flex-1 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm min-w-0 space-y-3 relative",
        )(
            Div(cls="flex items-center space-x-2.5 text-xs text-slate-600 dark:text-slate-400 font-medium")(
                Div(cls="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"),
                Span("Re-evaluating local knowledge base & generating fresh answer..."),
            )
        ),
    )

    remaining_messages = store.get_chat_messages(chat_id)
    chats = store.list_chats()

    turns = []
    i = 0
    while i < len(remaining_messages):
        msg = remaining_messages[i]
        if msg.id == message_id:
            turns.append(
                Div(id=f"turn-{msg.id}", cls="space-y-3")(
                    UserMessageBubble(msg, chat_id),
                    assistant_placeholder,
                )
            )
            break
        elif msg.role == "user":
            next_msg = remaining_messages[i + 1] if (i + 1 < len(remaining_messages) and remaining_messages[i + 1].role == "assistant") else None
            turns.append(ChatMessageTurn(msg, next_msg, chat_id))
            i += 2 if next_msg else 1
        else:
            turns.append(AssistantMessageBubble(msg, chat_id))
            i += 1

    return (
        ChatSidebar(chats=chats, active_chat_id=chat_id, hx_swap_oob="true"),
        Div(
            id="chat-main-area",
            cls="flex-1 flex flex-col min-h-0 h-full p-4 md:px-6 md:py-4 overflow-y-scroll scroll-smooth",
        )(
            Div(cls="sticky top-0 z-10 bg-base-200/95 backdrop-blur-sm pt-1 pb-3 mb-2 flex-shrink-0")(
                Div(cls="flex flex-wrap items-center justify-between gap-3 p-3 bg-base-100 border border-base-300 rounded-xl shadow-sm text-xs w-full")(
                    Div(cls="flex items-center gap-2")(
                        Span(cls="font-semibold text-base-content/70 flex items-center gap-1.5")(
                            Span("🔍"),
                            "Target Scope:"
                        ),
                        Select(id="doc-type-filter", name="doc_type_filter", cls="uk-select uk-select-sm text-xs rounded-lg w-44 bg-base-200 border-base-300")(
                            Option(value="all", selected=(doc_type_filter == "all"))("All Sources (Auto-route)"),
                            Option(value="code", selected=(doc_type_filter == "code"))("Source Code Only"),
                            Option(value="ticket", selected=(doc_type_filter == "ticket"))("ADO Work Items / Bugs"),
                            Option(value="confluence", selected=(doc_type_filter == "confluence"))("Confluence Wiki Pages"),
                        ),
                    ),
                    Div(cls="flex items-center gap-2")(
                        Span(cls="font-semibold text-base-content/70 flex items-center gap-1.5")(
                            Span("🏷️"),
                            "Sprint Filter:"
                        ),
                        Input(
                            id="sprint-filter",
                            name="sprint_filter",
                            type="text",
                            value=sprint_filter or "",
                            placeholder="e.g. Sprint 42 (optional)",
                            cls="uk-input uk-input-sm text-xs rounded-lg w-48 bg-base-200 border-base-300",
                        ),
                    ),
                ),
            ),
            Div(id="chat-history", cls="space-y-4 pb-6 flex-1")(*turns),
            Div(cls="sticky bottom-0 z-20 bg-base-200/95 backdrop-blur-md pt-2 pb-4 flex-shrink-0")(
                Form(
                    id="chat-form",
                    hx_post="/api/chat",
                    hx_target="#chat-history",
                    hx_swap="beforeend",
                    hx_include="#doc-type-filter, #sprint-filter, #active-chat-id-input",
                    cls="w-full bg-base-100 border border-base-300 rounded-2xl shadow-md p-2.5 focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/20 transition-all",
                )(
                    Input(type="hidden", id="active-chat-id-input", name="chat_id", value=chat_id),
                    Div(cls="flex flex-col gap-1.5")(
                        Textarea(
                            id="query-input",
                            name="query",
                            required=True,
                            rows=1,
                            placeholder="Ask a technical question about your code, boards, or wiki... (Enter to send, Shift+Enter for newline)",
                            cls="w-full bg-transparent border-0 focus:outline-none focus:ring-0 text-sm text-base-content placeholder:text-base-content/50 resize-none py-1.5 px-2.5 leading-normal min-h-[44px] max-h-[160px]",
                        ),
                        Div(cls="flex items-center justify-between pt-1.5 border-t border-base-200 text-xs text-base-content/60")(
                            Div(cls="flex items-center gap-1.5")(
                                Span(cls="text-[11px] hidden sm:inline")("Press"),
                                Span(cls="px-1.5 py-0.5 bg-base-200 border border-base-300 rounded text-[10px] font-mono text-base-content/80")("Enter ↵"),
                                Span(cls="text-[11px] hidden sm:inline")("to send"),
                            ),
                            Button(
                                type="submit",
                                cls="uk-button uk-button-primary uk-button-sm rounded-xl px-4 py-1 flex items-center gap-1.5 font-medium shadow-sm hover:shadow cursor-pointer",
                            )(
                                Span("Ask"),
                                Span(cls="text-xs font-bold")("→"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


@rt("/api/chats/{chat_id}/messages/{message_id}", methods=["DELETE"])
def delete_message(chat_id: str, message_id: str):
    """Deletes a message or message pair from DuckDB and removes it from the UI."""
    msg = store.get_message(message_id)
    if msg and msg.role == "user":
        store.delete_message_pair(chat_id, message_id)
    else:
        store.delete_message(message_id)
    return ""


@rt("/api/chats/{chat_id}/assistant-messages/{message_id}/edit-form", methods=["GET"])
def get_edit_assistant_message_form(chat_id: str, message_id: str):
    """Returns inline edit form replacing the assistant message bubble."""
    msg = store.get_message(message_id)
    if not msg:
        return Div("Message not found", cls="text-xs text-rose-500")
    return EditAssistantMessageForm(chat_id=chat_id, message_id=message_id, current_content=msg.content)


@rt("/api/chats/{chat_id}/assistant-messages/{message_id}/cancel-edit", methods=["GET"])
def get_cancel_edit_assistant(chat_id: str, message_id: str):
    """Cancels assistant editing and restores original assistant bubble."""
    msg = store.get_message(message_id)
    if not msg:
        return Div()
    return AssistantMessageBubble(msg=msg, chat_id=chat_id)


@rt("/api/chats/{chat_id}/assistant-messages/{message_id}/edit", methods=["POST"])
def post_edit_assistant_message(chat_id: str, message_id: str, content: str):
    """Saves edited content to DuckDB and re-renders assistant message bubble."""
    clean_content = content.strip()
    store.update_message_content(message_id, clean_content)
    store.touch_chat(chat_id)
    msg = store.get_message(message_id)
    if not msg:
        return Div("Message not found", cls="text-xs text-rose-500")
    return AssistantMessageBubble(msg=msg, chat_id=chat_id)


@rt("/api/chat", methods=["POST"])
async def post_chat(
    query: str,
    chat_id: Optional[str] = None,
    doc_type_filter: Optional[str] = "all",
    sprint_filter: Optional[str] = None,
):
    """
    Handles user chat submission:
    Creates or resumes session in DuckDB, persists user message,
    and returns user bubble + SSE placeholder for assistant response.
    """
    clean_query = query.strip()
    if not chat_id or not chat_id.strip():
        chat_id = str(uuid.uuid4())[:8]
        title = clean_query.replace("\n", " ")[:60].strip()
        store.create_chat(chat_id=chat_id, title=title)

    user_msg_id = str(uuid.uuid4())[:8]
    user_record = store.add_chat_message(
        message_id=user_msg_id,
        chat_id=chat_id,
        role="user",
        content=clean_query,
    )

    stream_id = str(uuid.uuid4())[:8]
    assistant_msg_id = str(uuid.uuid4())[:8]
    encoded_query = urllib.parse.quote(clean_query)
    encoded_doc_filter = urllib.parse.quote(doc_type_filter or "all")
    encoded_sprint = urllib.parse.quote(sprint_filter.strip() if sprint_filter else "")

    stream_url = f"/api/chat/stream/{stream_id}?chat_id={chat_id}&assistant_msg_id={assistant_msg_id}&q={encoded_query}&doc_type={encoded_doc_filter}&sprint={encoded_sprint}"
    assistant_placeholder = Div(
        id=f"assistant-bubble-{assistant_msg_id}",
        hx_ext="sse",
        sse_connect=stream_url,
        sse_swap="message",
        sse_close="close",
        hx_target=f"#response-box-{stream_id}",
        cls="group flex items-start space-x-3 w-full my-2",
    )(
        Div(cls="w-8 h-8 rounded-full bg-slate-800 dark:bg-slate-700 text-white flex-shrink-0 flex items-center justify-center font-black font-mono text-[11px] shadow-sm mt-0.5")("AI"),
        Div(
            id=f"response-box-{stream_id}",
            cls="flex-1 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm min-w-0 space-y-3 relative",
        )(
            Div(cls="flex items-center space-x-2.5 text-xs text-slate-600 dark:text-slate-400 font-medium")(
                Div(cls="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"),
                Span("Searching local knowledge base & generating grounded response..."),
            )
        ),
    )

    turn_element = Div(id=f"turn-{user_msg_id}", cls="space-y-3")(
        UserMessageBubble(user_record, chat_id),
        assistant_placeholder,
    )

    chats = store.list_chats()
    sidebar_oob = ChatSidebar(chats=chats, active_chat_id=chat_id, hx_swap_oob="true")
    chat_id_input_oob = Input(type="hidden", id="active-chat-id-input", name="chat_id", value=chat_id, hx_swap_oob="true")

    return turn_element, sidebar_oob, chat_id_input_oob


@rt("/api/chat/stream/{stream_id}")
async def get_chat_stream(
    stream_id: str,
    q: str,
    chat_id: Optional[str] = None,
    assistant_msg_id: Optional[str] = None,
    doc_type: Optional[str] = "all",
    sprint: Optional[str] = None,
):
    """
    SSE endpoint:
    Streams token-by-token generation from Ollama with sliding-window context.
    On completion, persists assistant message and citations in DuckDB.
    """
    query = urllib.parse.unquote(q)
    doc_filter = urllib.parse.unquote(doc_type) if doc_type != "all" else None
    sprint_f = urllib.parse.unquote(sprint) if sprint else None
    asst_id = assistant_msg_id or str(uuid.uuid4())[:8]

    async def event_generator():
        try:
            retrieved_docs, token_stream = await rag_pipeline.answer_stream(
                query=query,
                chat_id=chat_id,
                doc_type_filter=doc_filter,
                sprint_filter=sprint_f,
            )

            accumulated_response = ""
            async for token in token_stream:
                accumulated_response += token
                html_body = mistletoe.markdown(accumulated_response)
                formatted_body = format_inline_citations(html_body)

                streaming_element = Div(cls="space-y-3")(
                    Div(cls="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-slate-800/80 -mt-1")(
                        Div(cls="flex items-center gap-1.5 text-xs font-semibold text-slate-500 dark:text-slate-400 font-mono")(
                            Span("🤖", cls="text-xs"),
                            Span("AI Response"),
                        ),
                        Div(cls="flex items-center gap-1.5 text-[11px] text-blue-600 dark:text-blue-400 font-mono font-medium")(
                            Div(cls="w-1.5 h-1.5 rounded-full bg-blue-600 animate-pulse"),
                            Span("Generating..."),
                        ),
                    ),
                    Div(cls="prose prose-sm max-w-none text-slate-800 dark:text-slate-100 leading-relaxed font-normal")(
                        NotStr(formatted_body),
                        Span(cls="inline-block w-2 h-4 ml-1 bg-blue-600 animate-pulse align-middle"),
                    ),
                )
                yield sse_message(streaming_element)

            # Persist assistant reply with serialized citations to DuckDB
            serialized_citations = serialize_citations(retrieved_docs)
            if chat_id:
                store.add_chat_message(
                    message_id=asst_id,
                    chat_id=chat_id,
                    role="assistant",
                    content=accumulated_response,
                    citations=serialized_citations,
                )

            final_html_body = mistletoe.markdown(accumulated_response)
            formatted_final_body = format_inline_citations(final_html_body)
            citations_component = CitationDrawer(retrieved_docs)

            final_element = Div(cls="space-y-3")(
                Div(cls="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-slate-800/80 -mt-1")(
                    Div(cls="flex items-center gap-1.5 text-xs font-semibold text-slate-500 dark:text-slate-400 font-mono")(
                        Span("🤖", cls="text-xs"),
                        Span("AI Response"),
                    ),
                    Div(cls="flex items-center gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity")(
                        Button(
                            hx_get=f"/api/chats/{chat_id}/assistant-messages/{asst_id}/edit-form",
                            hx_target=f"#assistant-bubble-{asst_id}",
                            hx_swap="outerHTML",
                            title="Edit AI response",
                            cls="px-2 py-1 bg-base-200 hover:bg-base-300 border border-base-300 rounded-lg text-slate-600 dark:text-slate-300 font-medium text-[11px] shadow-2xs hover:shadow-xs transition-all cursor-pointer flex items-center gap-1",
                        )(
                            Span("✏️", cls="text-[10px]"),
                            Span("Edit"),
                        ),
                        Button(
                            hx_delete=f"/api/chats/{chat_id}/messages/{asst_id}",
                            hx_confirm="Are you sure you want to delete this AI response?",
                            hx_target=f"#assistant-bubble-{asst_id}",
                            hx_swap="outerHTML",
                            title="Delete response",
                            cls="px-1.5 py-1 bg-base-200 hover:bg-rose-100 dark:hover:bg-rose-950/60 border border-base-300 rounded-lg text-slate-500 hover:text-rose-700 font-medium text-[11px] shadow-2xs hover:shadow-xs transition-all cursor-pointer flex items-center",
                        )(
                            Span("🗑️", cls="text-[10px]"),
                        ),
                    ) if chat_id else None,
                ),
                Div(cls="prose prose-sm max-w-none text-slate-800 dark:text-slate-100 leading-relaxed font-normal")(
                    NotStr(formatted_final_body)
                ),
                citations_component,
            )
            yield sse_message(final_element)
            yield "event: close\ndata: finished\n\n"

        except Exception as e:
            error_element = Div(
                Div(cls="uk-alert uk-alert-danger text-xs p-3 rounded font-mono")(
                    f"Generation error: {str(e)}. Ensure Ollama is running and models are pulled."
                )
            )
            yield sse_message(error_element)
            yield "event: close\ndata: error\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    # Memory friendly default: 1 worker process
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, workers=1)
