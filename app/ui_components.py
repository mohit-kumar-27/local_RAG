"""
MonsterUI & FastHTML FT components for the Local RAG System.
All components are pure Python FT trees—no Jinja2 templates are used.
"""

from datetime import datetime
import html
import re
from typing import Any, Dict, List, Optional, Tuple

from fasthtml.common import (
    A, Button, Details, Div, Form, H1, H2, H3, H4, Input, Label, Li,
    NotStr, Option, P, Pre, Select, Span, Summary, Textarea, Ul, to_xml
)
import mistletoe
import monsterui.all as ui

from config import LOW_RAM_MODE, get_active_llm_model
from ingestion.base import Document
from rag.duckdb_store import ChatMessageRecord, ChatSession


def AppHeader(ollama_connected: bool = True, current_model: str = "", active_tab: str = "chat", cls: Optional[str] = None, **kwargs):
    """Header bar with title, confidential security badge, and RAM mode indicators."""
    status_color = "uk-badge-success" if ollama_connected else "uk-badge-danger"
    status_text = "Ollama Active" if ollama_connected else "Ollama Disconnected"

    ram_mode_label = "3B Low-RAM" if LOW_RAM_MODE else "8B Standard"
    ram_btn_style = "uk-button-secondary" if LOW_RAM_MODE else "uk-button-default"

    base_cls = "flex flex-col md:flex-row justify-between items-start md:items-center py-4 px-6 bg-base-200 border-b border-base-300 gap-4 flex-shrink-0"
    if cls:
        base_cls = f"{base_cls} {cls}"

    return Div(
        id=kwargs.pop("id", "app-header"),
        cls=base_cls,
        **kwargs,
    )(
        Div(cls="flex items-center space-x-3")(
            Div(cls="p-2 bg-primary text-primary-content rounded-lg font-mono font-bold text-xl")("RAG"),
            Div(
                H1(cls="text-xl font-bold tracking-tight")("Local Confidential RAG & Chatbot"),
                P(cls="text-xs text-base-content/70")("GitHub · Azure DevOps Boards & Repos · Confluence · DuckDB · Ollama"),
            ),
        ),
        Div(cls="flex flex-wrap items-center gap-2")(
            Span(cls="uk-badge uk-badge-primary font-mono text-xs")("Confidential · Zero Telemetry"),
            Span(cls=f"uk-badge {status_color} font-mono text-xs")(status_text),
            Span(cls="uk-badge uk-badge-secondary font-mono text-xs")(f"Model: {current_model or get_active_llm_model()}"),
            Form(action="/api/toggle_ram_mode", method="post", hx_post="/api/toggle_ram_mode", hx_target="#main-content")(
                Input(type="hidden", name="current_tab", value=active_tab),
                Button(type="submit", cls=f"uk-button {ram_btn_style} uk-button-xs font-mono")(
                    f"RAM: {ram_mode_label} (Toggle)"
                )
            ),
        ),
    )


def TabNavigation(active_tab: str = "chat", **kwargs):
    """
    Segmented pill navigation for Ingestion and Chatbot views.
    Distinct, high-contrast, and completely consistent across page reloads and HTMX swaps.
    """
    is_chat = (active_tab == "chat")
    is_ingest = (active_tab == "ingest")

    chat_btn_cls = (
        "bg-primary text-primary-content font-bold shadow-sm shadow-primary/40 rounded-lg px-4 py-2 "
        "flex items-center gap-2 transition-all duration-150 transform scale-[1.02] cursor-default"
        if is_chat else
        "text-base-content/70 hover:text-base-content hover:bg-base-300/60 font-medium rounded-lg px-4 py-2 "
        "flex items-center gap-2 transition-all duration-150"
    )

    ingest_btn_cls = (
        "bg-primary text-primary-content font-bold shadow-sm shadow-primary/40 rounded-lg px-4 py-2 "
        "flex items-center gap-2 transition-all duration-150 transform scale-[1.02] cursor-default"
        if is_ingest else
        "text-base-content/70 hover:text-base-content hover:bg-base-300/60 font-medium rounded-lg px-4 py-2 "
        "flex items-center gap-2 transition-all duration-150"
    )

    chat_dot = Span(cls="w-2 h-2 rounded-full bg-white animate-pulse") if is_chat else None
    ingest_dot = Span(cls="w-2 h-2 rounded-full bg-white animate-pulse") if is_ingest else None

    return Div(
        id="main-tab-nav",
        cls="bg-base-100 border-b border-base-300 px-6 py-2.5 flex items-center justify-between flex-shrink-0",
        **kwargs,
    )(
        Div(cls="flex items-center gap-3")(
            Span(cls="text-xs font-semibold uppercase tracking-wider text-base-content/50")("Views:"),
            Div(cls="bg-base-200 p-1 rounded-xl inline-flex border border-base-300 shadow-inner gap-1")(
                A(
                    href="/?tab=chat",
                    hx_get="/tab/chat",
                    hx_target="#tab-content",
                    hx_swap="outerHTML",
                    hx_push_url="true",
                    cls=chat_btn_cls,
                )(
                    Span(cls="text-base")("💬"),
                    Span("Ask Chatbot"),
                    chat_dot,
                ),
                A(
                    href="/?tab=ingest",
                    hx_get="/tab/ingest",
                    hx_target="#tab-content",
                    hx_swap="outerHTML",
                    hx_push_url="true",
                    cls=ingest_btn_cls,
                )(
                    Span(cls="text-base")("📥"),
                    Span("Ingest Sources"),
                    ingest_dot,
                ),
            ),
        ),
        # Current active page breadcrumb indicator
        Div(cls="hidden sm:flex items-center gap-2 text-xs")(
            Span(cls="text-base-content/50 font-medium")("Current Page:"),
            Span(cls="px-2.5 py-1 bg-primary/10 text-primary border border-primary/20 rounded-md font-mono font-semibold flex items-center gap-1.5")(
                Span(cls="w-1.5 h-1.5 rounded-full bg-primary"),
                "💬 Chatbot & Context Q&A" if is_chat else "📥 Source Ingestion & Sync"
            ),
        ),
    )


def CollectionStatsCard(stats: Dict[str, Any], cls: Optional[str] = None, **kwargs):
    """Renders collection statistics card with document counts and disk usage."""
    by_type = stats.get("by_type", {})
    code_count = by_type.get("code", 0)
    ticket_count = by_type.get("ticket", 0)
    confluence_count = by_type.get("confluence", 0)
    total_docs = stats.get("total_documents", 0)
    disk_mb = stats.get("disk_size_mb", 0.0)

    base_cls = "card bg-base-100 shadow-sm border border-slate-200 dark:border-slate-800 p-6 space-y-4"
    if cls:
        base_cls = f"{base_cls} {cls}"

    return Div(
        id=kwargs.pop("id", "collection-stats-card"),
        cls=base_cls,
        **kwargs,
    )(
        Div(cls="flex flex-wrap justify-between items-center gap-3 pb-2 border-b border-slate-200 dark:border-slate-800")(
            Div(cls="flex items-center gap-2")(
                Span(cls="text-base")("📊"),
                H3(cls="text-base font-bold text-slate-900 dark:text-white tracking-tight")("Local Knowledge Base Stats"),
            ),
            Div(cls="flex items-center gap-2.5")(
                Button(
                    hx_get="/api/stats",
                    hx_target="#collection-stats-card",
                    hx_swap="outerHTML",
                    cls="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg shadow-2xs hover:shadow-xs transition-all active:scale-95 cursor-pointer",
                )(
                    Span("🔄", cls="text-xs"),
                    Span("Refresh"),
                ),
                Button(
                    hx_post="/api/clear",
                    hx_confirm="Are you sure you want to delete all indexed data in DuckDB?",
                    hx_target="#collection-stats-card",
                    hx_swap="outerHTML",
                    cls="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-rose-700 dark:text-rose-200 bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-800 hover:bg-rose-100 dark:hover:bg-rose-900/60 rounded-lg shadow-2xs hover:shadow-xs transition-all active:scale-95 cursor-pointer",
                )(
                    Span("🗑️", cls="text-xs"),
                    Span("Clear DB"),
                ),
            ),
        ),
        Div(cls="grid grid-cols-2 md:grid-cols-4 gap-4 text-center")(
            Div(cls="p-4 bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl shadow-2xs")(
                Div(cls="text-2xl md:text-3xl font-black font-mono text-slate-900 dark:text-white tracking-tight")(str(total_docs)),
                Div(cls="text-xs font-semibold text-slate-600 dark:text-slate-400 mt-1")("Total Chunks"),
            ),
            Div(cls="p-4 bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl shadow-2xs")(
                Div(cls="text-2xl md:text-3xl font-black font-mono text-slate-900 dark:text-white tracking-tight")(str(code_count)),
                Div(cls="text-xs font-semibold text-slate-600 dark:text-slate-400 mt-1")("Code Chunks"),
            ),
            Div(cls="p-4 bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl shadow-2xs")(
                Div(cls="text-2xl md:text-3xl font-black font-mono text-slate-900 dark:text-white tracking-tight")(str(ticket_count)),
                Div(cls="text-xs font-semibold text-slate-600 dark:text-slate-400 mt-1")("ADO Work Items"),
            ),
            Div(cls="p-4 bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl shadow-2xs")(
                Div(cls="text-2xl md:text-3xl font-black font-mono text-slate-900 dark:text-white tracking-tight")(str(confluence_count)),
                Div(cls="text-xs font-semibold text-slate-600 dark:text-slate-400 mt-1")("Confluence Wiki"),
            ),
        ),
        Div(cls="pt-1 text-xs text-slate-500 dark:text-slate-400 flex flex-wrap justify-between items-center gap-2")(
            Span(cls="flex items-center gap-1.5")(
                Span("💾"),
                Span(f"Disk storage: {disk_mb} MB"),
            ),
            Span(cls="flex items-center gap-1.5 font-mono text-[11px] bg-slate-100 dark:bg-slate-800 px-2.5 py-1 rounded-md border border-slate-200 dark:border-slate-700")(
                Span("🦆"),
                Span("DuckDB: Vectors (Cosine) + BM25 FTS"),
            ),
        ),
    )


def IngestionTab(stats: Dict[str, Any], cls: Optional[str] = None, **kwargs):
    """Tab 1: Ingest Sources view with form, live SSE progress container, and stats."""
    base_cls = "max-w-5xl mx-auto py-6 px-4 space-y-6 pb-16 w-full"
    if cls:
        base_cls = f"{base_cls} {cls}"
    return Div(
        id=kwargs.pop("id", "ingestion-tab-container"),
        cls=base_cls,
        **kwargs,
    )(
        # Stats summary
        CollectionStatsCard(stats),

        # Ingestion form card
        Div(cls="card bg-base-100 shadow border border-base-200 p-6")(
            H3(cls="text-lg font-bold mb-1")("Ingest New Source"),
            P(cls="text-sm text-base-content/70 mb-4")(
                "Paste GitHub repositories, Azure DevOps Boards / Repos, or Atlassian Confluence spaces. "
                "The pipeline uses shallow clones, AST parsing, and SHA256 deduplication to ensure maximum efficiency."
            ),
            Form(
                hx_post="/api/ingest",
                hx_target="#ingest-progress-container",
                hx_swap="innerHTML",
                cls="space-y-4",
            )(
                Div(cls="grid grid-cols-1 md:grid-cols-3 gap-4")(
                    Div(
                        Label(cls="label text-xs font-semibold uppercase text-base-content/70")("Source Type"),
                        Select(name="source_type", cls="uk-select w-full")(
                            Option(value="github")("GitHub Repository (ghapi + git clone)"),
                            Option(value="ado_board")("Azure DevOps Board (Work Items / Sprints)"),
                            Option(value="ado_repo")("Azure DevOps Git Repo"),
                            Option(value="confluence")("Atlassian Confluence Space/Page"),
                        ),
                    ),
                    Div(cls="md:col-span-2")(
                        Label(cls="label text-xs font-semibold uppercase text-base-content/70")("Source URL / Space Key"),
                        Input(
                            name="url",
                            type="text",
                            required=True,
                            placeholder="e.g. https://github.com/fastai/fastcore or https://dev.azure.com/org/proj/_boards",
                            cls="uk-input w-full",
                        ),
                    ),
                ),
                Div(cls="grid grid-cols-1 md:grid-cols-2 gap-4")(
                    Div(
                        Label(cls="label text-xs font-semibold uppercase text-base-content/70")("Branch / Sprint / Filter (Optional)"),
                        Input(
                            name="branch_or_filter",
                            type="text",
                            placeholder="e.g. main, or Sprint 42",
                            cls="uk-input w-full",
                        ),
                    ),
                    Div(
                        Label(cls="label text-xs font-semibold uppercase text-base-content/70")("Personal Access Token Override (Optional)"),
                        Input(
                            name="token_override",
                            type="password",
                            placeholder="Leave blank to use .env credentials",
                            cls="uk-input w-full",
                        ),
                    ),
                ),
                Div(cls="flex justify-end pt-2")(
                    Button(
                        type="submit",
                        cls="uk-button uk-button-primary",
                    )("Start Ingestion"),
                ),
            ),
        ),

        # Ingestion Progress SSE target
        Div(id="ingest-progress-container", cls="min-h-[50px]"),
    )


def IngestProgressSSEComponent(job_id: str):
    """Component that initiates SSE connection to stream live ingestion progress."""
    return Div(
        hx_ext="sse",
        sse_connect=f"/api/ingest/stream/{job_id}",
        sse_swap="message",
        sse_close="close",
        cls="card bg-base-100 shadow border border-base-200 p-5",
    )(
        Div(cls="flex items-center space-x-3 mb-2")(
            Div(cls="animate-spin h-5 w-5 border-2 border-primary border-t-transparent rounded-full"),
            Span(cls="font-semibold text-sm")("Initializing ingestion worker..."),
        ),
        ui.Progress(value=5, max=100, cls="w-full h-2"),
    )


def IngestProgressUpdateCard(
    job_id: str,
    stage: str,
    progress: int,
    status: str,
    logs: List[str],
    error_message: Optional[str] = None,
):
    """Rendered inside SSE event to update progress and logs in real time."""
    status_badge_cls = {
        "running": "uk-badge-primary",
        "completed": "uk-badge-success",
        "failed": "uk-badge-danger",
        "queued": "uk-badge-warning",
    }.get(status, "uk-badge-default")

    log_content = "\n".join(logs[-8:]) if logs else "No logs yet."

    return Div(cls="card bg-base-100 shadow border border-base-200 p-5 space-y-3")(
        Div(cls="flex justify-between items-center")(
            Div(cls="flex items-center space-x-2")(
                Span(cls=f"uk-badge {status_badge_cls} font-mono text-xs uppercase")(status),
                Span(cls="text-sm font-semibold text-base-content")(stage),
            ),
            Span(cls="font-mono text-xs font-bold text-primary")(f"{progress}%"),
        ),
        ui.Progress(value=progress, max=100, cls="w-full h-2"),
        Div(cls="uk-alert uk-alert-danger text-xs p-3 rounded-lg font-mono")(error_message) if error_message else None,
        Div(cls="bg-base-300 rounded-lg p-3")(
            Pre(cls="text-xs font-mono text-base-content/80 whitespace-pre-wrap overflow-x-auto max-h-36")(
                log_content
            )
        ),
        (
            Div(cls="pt-2 flex justify-end")(
                Button(
                    hx_get="/api/stats",
                    hx_target="#collection-stats-card",
                    hx_swap="outerHTML",
                    cls="uk-button uk-button-primary uk-button-sm",
                )("Update Stats")
            )
            if status == "completed"
            else None
        ),
    )


def CitationDrawer(documents_with_scores: List[Tuple[Document, float]]):
    """
    High-contrast, accessible citation sources drawer displaying collapsible
    chunks and metadata used to generate the answer.
    Uses native Details/Summary styled with robust Tailwind classes to prevent
    theme background hijacks (e.g. blue-on-blue text).
    """
    if not documents_with_scores:
        return Div()

    citation_cards = []
    for idx, (doc, score) in enumerate(documents_with_scores, start=1):
        citation = doc.get_citation_tag()

        # Dedicated, accessible high-contrast badge styles per source type
        badge_style = {
            "code": "bg-sky-100 text-sky-900 border-sky-300 dark:bg-sky-950 dark:text-sky-200 dark:border-sky-800",
            "ticket": "bg-emerald-100 text-emerald-900 border-emerald-300 dark:bg-emerald-950 dark:text-emerald-200 dark:border-emerald-800",
            "confluence": "bg-purple-100 text-purple-900 border-purple-300 dark:bg-purple-950 dark:text-purple-200 dark:border-purple-800",
        }.get(doc.doc_type, "bg-slate-100 text-slate-900 border-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700")

        summary_component = Summary(
            cls=(
                "flex flex-wrap items-center justify-between gap-2.5 p-3.5 cursor-pointer select-none list-none "
                "[&::-webkit-details-marker]:hidden bg-slate-50 dark:bg-slate-800/90 hover:bg-slate-100/90 "
                "dark:hover:bg-slate-800 transition-colors rounded-xl border border-slate-200 dark:border-slate-700 "
                "group-open:rounded-b-none group-open:border-b-transparent"
            )
        )(
            Div(cls="flex items-center gap-2.5 min-w-0 flex-1")(
                Span(
                    cls=f"px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-wider font-mono border shadow-2xs flex-shrink-0 {badge_style}"
                )(doc.doc_type),
                Span(
                    cls="font-bold font-mono text-xs text-slate-900 dark:text-white truncate"
                )(citation),
            ),
            Div(cls="flex items-center gap-3 text-xs font-mono flex-shrink-0")(
                Span(cls="text-slate-600 dark:text-slate-300 font-semibold text-[11px]")(
                    f"Relevance: {score:.3f}"
                ),
                Span(
                    cls="text-slate-400 dark:text-slate-400 text-xs transition-transform duration-200 group-open:rotate-180"
                )("▼"),
            ),
        )

        body_content = Div(
            cls=(
                "p-4 space-y-3 font-mono text-xs bg-white dark:bg-slate-900 border-x border-b "
                "border-slate-200 dark:border-slate-700 rounded-b-xl shadow-xs"
            )
        )(
            Div(
                cls="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-100 dark:border-slate-800 text-[11px]"
            )(
                Span(cls="font-bold text-slate-700 dark:text-slate-300")("Source URL:"),
                A(
                    href=doc.source_url,
                    target="_blank",
                    rel="noopener noreferrer",
                    cls="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline font-medium truncate max-w-lg",
                )(doc.source_url),
            ),
            (
                Div(cls="flex items-center gap-2 text-[11px]")(
                    Span(cls="font-bold text-slate-500 dark:text-slate-400")("File Path:"),
                    Span(cls="text-slate-800 dark:text-slate-200 font-medium truncate")(doc.file_path),
                )
                if doc.file_path
                else None
            ),
            (
                Div(cls="flex items-center gap-2 text-[11px]")(
                    Span(cls="font-bold text-slate-500 dark:text-slate-400")("Sprint:"),
                    Span(cls="text-slate-800 dark:text-slate-200 font-medium")(doc.sprint_id),
                )
                if doc.sprint_id
                else None
            ),
            (
                Div(cls="flex items-center gap-2 text-[11px]")(
                    Span(cls="font-bold text-slate-500 dark:text-slate-400")("Work Item ID:"),
                    Span(cls="text-slate-800 dark:text-slate-200 font-medium")(doc.work_item_id),
                )
                if doc.work_item_id
                else None
            ),
            Div(cls="pt-1")(
                Div(cls="flex items-center justify-between text-[11px] font-bold text-slate-700 dark:text-slate-300 mb-1.5")(
                    Span("Matched Source Snippet:"),
                    Span(f"{len(doc.content)} chars", cls="text-slate-400 dark:text-slate-500 font-mono text-[10px] font-normal"),
                ),
                Pre(
                    cls=(
                        "bg-[#0f172a] text-[#f8fafc] p-4 rounded-xl text-xs leading-relaxed font-mono "
                        "whitespace-pre-wrap overflow-x-auto max-h-64 border border-slate-800 "
                        "shadow-inner selection:bg-blue-600 selection:text-white"
                    )
                )(doc.content),
            ),
        )

        citation_cards.append(
            Details(cls="group mb-2.5")(
                summary_component,
                body_content,
            )
        )

    return Div(cls="mt-4 pt-3 border-t border-slate-200 dark:border-slate-800")(
        Div(cls="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2")(
            Span("📚", cls="text-sm"),
            Span(f"Retrieved Context Sources ({len(documents_with_scores)})"),
        ),
        Div(cls="space-y-1")(
            *citation_cards
        ),
    )


def format_relative_time(dt: Any) -> str:
    """Formats a datetime into a friendly relative timestamp."""
    if not dt:
        return ""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace(" ", "T"))
        except Exception:
            return dt[:10]
    now = datetime.now(dt.tzinfo) if getattr(dt, "tzinfo", None) else datetime.now()
    diff = now - dt
    secs = int(diff.total_seconds())
    if secs < 60:
        return "Just now"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days}d ago"
    return dt.strftime("%b %d")


def format_inline_citations(html_text: str) -> str:
    """Styles [Source: ...] inline citations as clean, readable high-contrast badges."""
    def _replacer(match):
        raw_label = match.group(1).strip()
        escaped_label = html.escape(raw_label)
        return (
            f'<span class="inline-flex items-center gap-1 px-2 py-0.5 mx-1 my-0.5 rounded-md text-[11px] font-mono font-bold '
            f'bg-blue-100 text-blue-900 border border-blue-300 dark:bg-blue-950/80 dark:text-blue-200 dark:border-blue-700 '
            f'shadow-2xs">📌 {escaped_label}</span>'
        )
    return re.sub(r'\[Source:\s*([^\]]+)\]', _replacer, html_text)


def serialize_citations(documents_with_scores: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
    """Serializes retrieved documents and their relevance scores to JSON-compatible dicts."""
    serialized = []
    for doc, score in documents_with_scores:
        serialized.append(
            {
                "id": doc.id,
                "content": doc.content,
                "source_url": doc.source_url,
                "file_path": doc.file_path,
                "doc_type": doc.doc_type,
                "sprint_id": doc.sprint_id,
                "work_item_id": doc.work_item_id,
                "score": score,
            }
        )
    return serialized


def deserialize_citations(citations_data: Optional[List[Dict[str, Any]]]) -> List[Tuple[Document, float]]:
    """Reconstructs Document objects and scores from stored JSON."""
    if not citations_data:
        return []
    docs_with_scores = []
    for item in citations_data:
        doc = Document(
            id=item.get("id", ""),
            content=item.get("content", ""),
            source_url=item.get("source_url", ""),
            file_path=item.get("file_path"),
            doc_type=item.get("doc_type", "code"),
            sprint_id=item.get("sprint_id"),
            work_item_id=item.get("work_item_id"),
        )
        score = float(item.get("score", 0.0))
        docs_with_scores.append((doc, score))
    return docs_with_scores


def ChatSidebar(chats: List[ChatSession], active_chat_id: Optional[str] = None, cls: Optional[str] = None, **kwargs):
    """Sidebar listing persisted chat sessions with New Chat button and delete controls."""
    chat_items = []
    for chat in chats:
        is_active = (chat.id == active_chat_id)
        active_cls = (
            "bg-primary/10 border-primary/40 text-primary dark:text-blue-400 font-semibold shadow-2xs"
            if is_active
            else "bg-base-100 hover:bg-base-200/80 border-base-300 text-base-content/80 hover:text-base-content"
        )
        rel_time = format_relative_time(chat.updated_at)

        item = Div(
            id=f"chat-item-{chat.id}",
            cls=f"group flex items-center justify-between p-2.5 rounded-xl border transition-all duration-150 text-xs {active_cls}",
        )(
            A(
                href=f"/?tab=chat&chat_id={chat.id}",
                hx_get=f"/api/chats/{chat.id}",
                hx_target="#chat-main-area",
                hx_push_url=f"/?tab=chat&chat_id={chat.id}",
                cls="flex-1 min-w-0 pr-2 flex flex-col gap-0.5 cursor-pointer",
            )(
                Span(cls="truncate text-xs font-medium tracking-tight")(chat.title or "Untitled Chat"),
                Span(cls="text-[10px] text-base-content/50 font-mono")(rel_time),
            ),
            Button(
                hx_delete=f"/api/chats/{chat.id}?active_chat_id={active_chat_id or ''}",
                hx_confirm="Are you sure you want to delete this entire chat session and all its messages?",
                hx_target="#chat-tab-container",
                title="Delete chat session",
                cls="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 text-slate-400 hover:text-rose-600 rounded-md hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-all cursor-pointer",
            )(
                Span("🗑️", cls="text-xs")
            ),
        )
        chat_items.append(item)

    empty_placeholder = (
        Div(cls="p-4 text-center text-xs text-base-content/50 space-y-1")(
            Div("💬", cls="text-lg"),
            P("No saved chats yet."),
            P("Start a conversation!"),
        )
        if not chat_items
        else None
    )

    base_cls = "w-full md:w-64 lg:w-72 flex-shrink-0 bg-base-100 border-r border-base-300 flex flex-col h-auto md:h-full min-h-0"
    if cls:
        base_cls = f"{base_cls} {cls}"

    return Div(
        id=kwargs.pop("id", "chat-sidebar"),
        cls=base_cls,
        **kwargs,
    )(
        Div(cls="p-3 border-b border-base-300 flex items-center justify-between gap-2 flex-shrink-0 bg-base-100/90")(
            Span(cls="text-xs font-bold uppercase tracking-wider text-base-content/70 flex items-center gap-1.5")(
                Span("💬"),
                "Sessions",
            ),
            A(
                href="/?tab=chat",
                hx_get="/api/chats/new",
                hx_target="#chat-main-area",
                hx_push_url="/?tab=chat",
                cls="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold text-primary bg-primary/10 hover:bg-primary/20 border border-primary/20 rounded-lg shadow-2xs transition-all active:scale-95 cursor-pointer",
            )(
                Span("+", cls="font-bold text-sm"),
                Span("New Chat"),
            ),
        ),
        Div(
            id="chat-sidebar-list",
            cls="flex-1 overflow-y-auto p-2.5 space-y-1.5 min-h-0",
        )(
            *chat_items,
            empty_placeholder,
        ),
    )


def UserMessageBubble(msg: ChatMessageRecord, chat_id: str, cls: Optional[str] = None, **kwargs):
    """Renders a user message bubble with inline Edit and Delete action controls."""
    base_cls = "flex items-start justify-end gap-2.5 w-full my-1.5"
    if cls:
        base_cls = f"{base_cls} {cls}"
    return Div(
        id=kwargs.pop("id", f"user-bubble-container-{msg.id}"),
        cls=base_cls,
        **kwargs,
    )(
        Div(cls="group relative flex items-start justify-end gap-2 max-w-2xl")(
            Div(
                cls="flex items-center gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity mt-1.5 flex-shrink-0 text-xs"
            )(
                Button(
                    hx_get=f"/api/chats/{chat_id}/messages/{msg.id}/edit-form",
                    hx_target=f"#user-bubble-container-{msg.id}",
                    hx_swap="outerHTML",
                    title="Edit message",
                    cls="px-2 py-1 bg-base-200 hover:bg-base-300 border border-base-300 rounded-lg text-slate-600 dark:text-slate-300 font-medium text-[11px] shadow-2xs hover:shadow-xs transition-all cursor-pointer flex items-center gap-1",
                )(
                    Span("✏️", cls="text-[10px]"),
                    Span("Edit"),
                ),
                Button(
                    hx_delete=f"/api/chats/{chat_id}/messages/{msg.id}",
                    hx_confirm="Are you sure you want to delete this question and its paired AI response?",
                    hx_target=f"#turn-{msg.id}",
                    hx_swap="outerHTML",
                    title="Delete question and response",
                    cls="px-1.5 py-1 bg-base-200 hover:bg-rose-100 dark:hover:bg-rose-950/60 border border-base-300 rounded-lg text-slate-500 hover:text-rose-700 font-medium text-[11px] shadow-2xs hover:shadow-xs transition-all cursor-pointer flex items-center",
                )(
                    Span("🗑️", cls="text-[10px]"),
                ),
            ),
            Div(
                cls="bg-blue-600 text-white rounded-2xl rounded-tr-xs px-4 py-2.5 shadow-sm max-w-xl text-sm leading-relaxed whitespace-pre-wrap break-words font-normal"
            )(msg.content.strip()),
        ),
        Div(
            cls="w-7 h-7 rounded-full bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 flex-shrink-0 flex items-center justify-center font-bold text-[10px] shadow-sm mt-0.5"
        )("YOU"),
    )


def AssistantMessageBubble(msg: ChatMessageRecord, chat_id: str, cls: Optional[str] = None, **kwargs):
    """Renders an assistant message bubble with formatted markdown, collapsible citations, and delete control."""
    raw_html = mistletoe.markdown(msg.content)
    formatted_html = format_inline_citations(raw_html)

    docs_with_scores = deserialize_citations(msg.citations)
    citations_component = CitationDrawer(docs_with_scores) if docs_with_scores else None

    base_cls = "group flex items-start space-x-3 w-full my-2"
    if cls:
        base_cls = f"{base_cls} {cls}"

    return Div(
        id=kwargs.pop("id", f"assistant-bubble-{msg.id}"),
        cls=base_cls,
        **kwargs,
    )(
        Div(
            cls="w-8 h-8 rounded-full bg-slate-800 dark:bg-slate-700 text-white flex-shrink-0 flex items-center justify-center font-black font-mono text-[11px] shadow-sm mt-0.5"
        )("AI"),
        Div(
            cls="flex-1 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm min-w-0 space-y-3 relative",
        )(
            Div(cls="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-slate-800/80 -mt-1")(
                Div(cls="flex items-center gap-1.5 text-xs font-semibold text-slate-500 dark:text-slate-400 font-mono")(
                    Span("🤖", cls="text-xs"),
                    Span("AI Response"),
                ),
                Div(cls="flex items-center gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity")(
                    Button(
                        hx_get=f"/api/chats/{chat_id}/assistant-messages/{msg.id}/edit-form",
                        hx_target=f"#assistant-bubble-{msg.id}",
                        hx_swap="outerHTML",
                        title="Edit AI response",
                        cls="px-2 py-1 bg-base-200 hover:bg-base-300 border border-base-300 rounded-lg text-slate-600 dark:text-slate-300 font-medium text-[11px] shadow-2xs hover:shadow-xs transition-all cursor-pointer flex items-center gap-1",
                    )(
                        Span("✏️", cls="text-[10px]"),
                        Span("Edit"),
                    ),
                    Button(
                        hx_delete=f"/api/chats/{chat_id}/messages/{msg.id}",
                        hx_confirm="Are you sure you want to delete this AI response?",
                        hx_target=f"#assistant-bubble-{msg.id}",
                        hx_swap="outerHTML",
                        title="Delete response",
                        cls="px-1.5 py-1 bg-base-200 hover:bg-rose-100 dark:hover:bg-rose-950/60 border border-base-300 rounded-lg text-slate-500 hover:text-rose-700 font-medium text-[11px] shadow-2xs hover:shadow-xs transition-all cursor-pointer flex items-center",
                    )(
                        Span("🗑️", cls="text-[10px]"),
                    ),
                ),
            ),
            Div(cls="prose prose-sm max-w-none text-slate-800 dark:text-slate-100 leading-relaxed font-normal")(
                NotStr(formatted_html)
            ),
            citations_component,
        ),
    )


def ChatMessageTurn(user_msg: ChatMessageRecord, assistant_msg: Optional[ChatMessageRecord], chat_id: str, cls: Optional[str] = None, **kwargs):
    """Groups a paired user prompt and assistant reply into a single turn container."""
    base_cls = "space-y-3"
    if cls:
        base_cls = f"{base_cls} {cls}"
    return Div(
        id=kwargs.pop("id", f"turn-{user_msg.id}"),
        cls=base_cls,
        **kwargs,
    )(
        UserMessageBubble(user_msg, chat_id),
        AssistantMessageBubble(assistant_msg, chat_id) if assistant_msg else None,
    )


def EditMessageForm(chat_id: str, message_id: str, current_content: str, cls: Optional[str] = None, **kwargs):
    """Inline edit form for updating an existing user prompt."""
    base_cls = "w-full my-2 flex justify-end"
    if cls:
        base_cls = f"{base_cls} {cls}"
    return Div(
        id=kwargs.pop("id", f"user-bubble-container-{message_id}"),
        cls=base_cls,
        **kwargs,
    )(
        Div(
            cls="w-full max-w-xl bg-base-100 border-2 border-primary/50 rounded-2xl p-4 shadow-md space-y-3"
        )(
            Div(cls="flex items-center justify-between text-xs font-semibold text-base-content/80")(
                Span(cls="flex items-center gap-1.5")(
                    Span("✏️"),
                    Span("Edit Prompt"),
                ),
                Span(cls="text-[10px] text-amber-600 dark:text-amber-400 font-mono bg-amber-50 dark:bg-amber-950/40 px-2 py-0.5 rounded border border-amber-200 dark:border-amber-800")(
                    "⚠️ Submitting prunes later turns"
                ),
            ),
            Form(
                hx_post=f"/api/chats/{chat_id}/messages/{message_id}/edit",
                hx_target="#chat-main-area",
                hx_include="#doc-type-filter, #sprint-filter",
                cls="space-y-2.5",
            )(
                Textarea(
                    name="new_content",
                    rows=2,
                    required=True,
                    cls="uk-textarea w-full text-sm rounded-xl border border-base-300 p-2.5 focus:border-primary focus:ring-1 focus:ring-primary/30 text-base-content",
                )(current_content.strip()),
                Div(cls="flex items-center justify-end gap-2 pt-1")(
                    Button(
                        type="button",
                        hx_get=f"/api/chats/{chat_id}/messages/{message_id}/cancel-edit",
                        hx_target=f"#user-bubble-container-{message_id}",
                        hx_swap="outerHTML",
                        cls="uk-button uk-button-default uk-button-xs rounded-lg cursor-pointer",
                    )("Cancel"),
                    Button(
                        type="submit",
                        cls="uk-button uk-button-primary uk-button-xs rounded-lg cursor-pointer shadow-sm",
                    )("Save & Regenerate"),
                ),
            ),
        )
    )


def EditAssistantMessageForm(chat_id: str, message_id: str, current_content: str, cls: Optional[str] = None, **kwargs):
    """Inline edit form for modifying an assistant response directly."""
    base_cls = "group flex items-start space-x-3 w-full my-2"
    if cls:
        base_cls = f"{base_cls} {cls}"

    return Div(
        id=kwargs.pop("id", f"assistant-bubble-{message_id}"),
        cls=base_cls,
        **kwargs,
    )(
        Div(
            cls="w-8 h-8 rounded-full bg-slate-800 dark:bg-slate-700 text-white flex-shrink-0 flex items-center justify-center font-black font-mono text-[11px] shadow-sm mt-0.5"
        )("AI"),
        Div(
            cls="flex-1 bg-white dark:bg-slate-900 border-2 border-primary/50 rounded-2xl p-5 shadow-md min-w-0 space-y-3"
        )(
            Div(cls="flex items-center justify-between text-xs font-semibold text-slate-800 dark:text-slate-200 border-b border-slate-100 dark:border-slate-800 pb-2")(
                Span(cls="flex items-center gap-1.5")(
                    Span("✏️"),
                    Span("Edit AI Response (Markdown)"),
                ),
                Span(cls="text-[10px] text-slate-500 dark:text-slate-400 font-mono")(
                    "Citations & context are preserved"
                ),
            ),
            Form(
                hx_post=f"/api/chats/{chat_id}/assistant-messages/{message_id}/edit",
                hx_target=f"#assistant-bubble-{message_id}",
                hx_swap="outerHTML",
                cls="space-y-3",
            )(
                Textarea(
                    name="content",
                    rows=8,
                    required=True,
                    cls="uk-textarea w-full text-xs font-mono rounded-xl border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 p-3.5 focus:border-primary focus:ring-1 focus:ring-primary/30 text-slate-900 dark:text-slate-100 resize-y leading-relaxed",
                )(current_content.strip()),
                Div(cls="flex items-center justify-end gap-2 pt-1")(
                    Button(
                        type="button",
                        hx_get=f"/api/chats/{chat_id}/assistant-messages/{message_id}/cancel-edit",
                        hx_target=f"#assistant-bubble-{message_id}",
                        hx_swap="outerHTML",
                        cls="uk-button uk-button-default uk-button-xs rounded-lg cursor-pointer",
                    )("Cancel"),
                    Button(
                        type="submit",
                        cls="uk-button uk-button-primary uk-button-xs rounded-lg cursor-pointer shadow-sm",
                    )("Save Changes"),
                ),
            ),
        ),
    )


def WelcomeMessage():
    """Default placeholder bubble shown in an empty conversation."""
    return Div(cls="flex items-start space-x-3")(
        Div(
            cls="w-9 h-9 rounded-full bg-primary text-primary-content flex-shrink-0 flex items-center justify-center font-bold text-xs shadow-sm"
        )("AI"),
        Div(cls="flex-1 bg-base-100 border border-base-300 rounded-2xl p-5 shadow-sm space-y-2")(
            P(cls="text-sm text-base-content font-medium")(
                "Hello! I am your strictly local, confidential RAG assistant."
            ),
            P(cls="text-xs text-base-content/70 leading-relaxed")(
                "I answer questions using only the codebases, Azure DevOps work items, and Confluence documentation indexed into your local DuckDB database. "
                "Every statement is grounded in local context with explicit inline citations."
            ),
            Div(cls="pt-2 flex flex-wrap gap-2")(
                Span(cls="text-[11px] bg-base-200 text-base-content/70 px-2.5 py-1 rounded-md border border-base-300")(
                    "💡 Example: 'How does authentication work in our services?'"
                ),
                Span(cls="text-[11px] bg-base-200 text-base-content/70 px-2.5 py-1 rounded-md border border-base-300")(
                    "💡 Example: 'What bugs are open in Sprint 42?'"
                ),
            ),
        ),
    )


def ChatMainArea(messages: Optional[List[ChatMessageRecord]] = None, active_chat_id: Optional[str] = None, cls: Optional[str] = None, **kwargs):
    """
    Main conversation area containing scope/sprint filters, message history timeline,
    and sticky prompt input bar.
    """
    turns = []
    if messages:
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == "user":
                next_msg = messages[i + 1] if (i + 1 < len(messages) and messages[i + 1].role == "assistant") else None
                turns.append(ChatMessageTurn(msg, next_msg, active_chat_id or msg.chat_id))
                i += 2 if next_msg else 1
            else:
                turns.append(AssistantMessageBubble(msg, active_chat_id or msg.chat_id))
                i += 1

    content_children = turns if turns else [WelcomeMessage()]

    base_cls = "flex-1 flex flex-col min-h-0 h-full p-4 md:px-6 md:py-4 overflow-y-scroll scroll-smooth"
    if cls:
        base_cls = f"{base_cls} {cls}"

    return Div(
        id=kwargs.pop("id", "chat-main-area"),
        cls=base_cls,
        **kwargs,
    )(
        Div(cls="sticky top-0 z-10 bg-base-200/95 backdrop-blur-sm pt-1 pb-3 mb-2 flex-shrink-0")(
            Div(cls="flex flex-wrap items-center justify-between gap-3 p-3 bg-base-100 border border-base-300 rounded-xl shadow-sm text-xs w-full")(
                Div(cls="flex items-center gap-2")(
                    Span(cls="font-semibold text-base-content/70 flex items-center gap-1.5")(
                        Span("🔍"),
                        "Target Scope:"
                    ),
                    Select(id="doc-type-filter", name="doc_type_filter", cls="uk-select uk-select-sm text-xs rounded-lg w-44 bg-base-200 border-base-300")(
                        Option(value="all")("All Sources (Auto-route)"),
                        Option(value="code")("Source Code Only"),
                        Option(value="ticket")("ADO Work Items / Bugs"),
                        Option(value="confluence")("Confluence Wiki Pages"),
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
                        placeholder="e.g. Sprint 42 (optional)",
                        cls="uk-input uk-input-sm text-xs rounded-lg w-48 bg-base-200 border-base-300",
                    ),
                ),
            ),
        ),

        Div(
            id="chat-history",
            cls="space-y-4 pb-6 flex-1",
        )(
            *content_children
        ),

        Div(cls="sticky bottom-0 z-20 bg-base-200/95 backdrop-blur-md pt-2 pb-4 flex-shrink-0")(
            Form(
                id="chat-form",
                hx_post="/api/chat",
                hx_target="#chat-history",
                hx_swap="beforeend",
                hx_include="#doc-type-filter, #sprint-filter, #active-chat-id-input",
                hx_on__after_request="""
                    const qInput = document.getElementById('query-input');
                    if (qInput) {
                        qInput.value = '';
                        qInput.style.height = '44px';
                    }
                    const mainArea = document.getElementById('chat-main-area');
                    if (mainArea) {
                        mainArea.scrollTop = mainArea.scrollHeight;
                    }
                """,
                cls="w-full bg-base-100 border border-base-300 rounded-2xl shadow-md p-2.5 focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/20 transition-all",
            )(
                Input(type="hidden", id="active-chat-id-input", name="chat_id", value=active_chat_id or ""),
                Div(cls="flex flex-col gap-1.5")(
                    Textarea(
                        id="query-input",
                        name="query",
                        required=True,
                        rows=1,
                        placeholder="Ask a technical question about your code, boards, or wiki... (Enter to send, Shift+Enter for newline)",
                        onkeydown="""
                            if (event.key === 'Enter' && !event.shiftKey) {
                                event.preventDefault();
                                if (this.value.trim().length > 0) {
                                    htmx.trigger(this.closest('form'), 'submit');
                                }
                            }
                        """,
                        oninput="this.style.height = 'auto'; this.style.height = Math.min(this.scrollHeight, 160) + 'px';",
                        cls="w-full bg-transparent border-0 focus:outline-none focus:ring-0 text-sm text-base-content placeholder:text-base-content/50 resize-none py-1.5 px-2.5 leading-normal min-h-[44px] max-h-[160px]",
                    ),
                    Div(cls="flex items-center justify-between pt-1.5 border-t border-base-200 text-xs text-base-content/60")(
                        Div(cls="flex items-center gap-1.5")(
                            Span(cls="text-[11px] hidden sm:inline")("Press"),
                            Span(cls="px-1.5 py-0.5 bg-base-200 border border-base-300 rounded text-[10px] font-mono text-base-content/80")("Enter ↵"),
                            Span(cls="text-[11px] hidden sm:inline")("to send,"),
                            Span(cls="px-1.5 py-0.5 bg-base-200 border border-base-300 rounded text-[10px] font-mono text-base-content/80")("Shift + Enter"),
                            Span(cls="text-[11px] hidden sm:inline")("for newline"),
                        ),
                        Div(cls="flex items-center gap-2")(
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


def ChatTab(chats: Optional[List[ChatSession]] = None, active_chat_id: Optional[str] = None, messages: Optional[List[ChatMessageRecord]] = None, cls: Optional[str] = None, **kwargs):
    """Tab 2: Ask Chatbot view with responsive session sidebar and conversation main area."""
    base_cls = "flex flex-col md:flex-row h-full min-h-full w-full"
    if cls:
        base_cls = f"{base_cls} {cls}"
    return Div(
        id=kwargs.pop("id", "chat-tab-container"),
        cls=base_cls,
        **kwargs,
    )(
        ChatSidebar(chats or [], active_chat_id=active_chat_id),
        ChatMainArea(messages=messages, active_chat_id=active_chat_id),
    )
