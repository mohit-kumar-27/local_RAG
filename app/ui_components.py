"""
MonsterUI & FastHTML FT components for the Local RAG System.
All components are pure Python FT trees—no Jinja2 templates are used.
"""

from typing import Any, Dict, List, Optional, Tuple
import html

from fasthtml.common import (
    A, Button, Div, Form, H1, H2, H3, H4, Input, Label, Li,
    Option, P, Pre, Select, Span, Ul, to_xml
)
import monsterui.all as ui

from config import LOW_RAM_MODE, get_active_llm_model
from ingestion.base import Document


def AppHeader(ollama_connected: bool = True, current_model: str = ""):
    """Header bar with title, confidential security badge, and RAM mode indicators."""
    status_color = "uk-badge-success" if ollama_connected else "uk-badge-danger"
    status_text = "Ollama Active" if ollama_connected else "Ollama Disconnected"

    ram_mode_label = "3B Low-RAM" if LOW_RAM_MODE else "8B Standard"
    ram_btn_style = "uk-button-secondary" if LOW_RAM_MODE else "uk-button-default"

    return Div(cls="flex flex-col md:flex-row justify-between items-start md:items-center py-4 px-6 bg-base-200 border-b border-base-300 gap-4")(
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
                Button(type="submit", cls=f"uk-button {ram_btn_style} uk-button-xs font-mono")(
                    f"RAM: {ram_mode_label} (Toggle)"
                )
            ),
        ),
    )


def TabNavigation(active_tab: str = "chat"):
    """Tab selector for Ingestion and Chatbot views."""
    ingest_active = "uk-active font-bold border-b-2 border-primary text-primary" if active_tab == "ingest" else "text-base-content/70 hover:text-base-content"
    chat_active = "uk-active font-bold border-b-2 border-primary text-primary" if active_tab == "chat" else "text-base-content/70 hover:text-base-content"

    return Div(cls="border-b border-base-300 bg-base-100 px-6 pt-2")(
        Ul(cls="flex space-x-8 text-sm")(
            Li(
                A(
                    href="/?tab=chat",
                    hx_get="/tab/chat",
                    hx_target="#tab-content",
                    hx_push_url="true",
                    cls=f"inline-flex items-center py-3 px-1 transition-all {chat_active}",
                )(
                    Span(cls="mr-2")("💬"),
                    "Ask Chatbot",
                )
            ),
            Li(
                A(
                    href="/?tab=ingest",
                    hx_get="/tab/ingest",
                    hx_target="#tab-content",
                    hx_push_url="true",
                    cls=f"inline-flex items-center py-3 px-1 transition-all {ingest_active}",
                )(
                    Span(cls="mr-2")("📥"),
                    "Ingest Sources",
                )
            ),
        )
    )


def CollectionStatsCard(stats: Dict[str, Any]):
    """Renders collection statistics card with document counts and disk usage."""
    by_type = stats.get("by_type", {})
    code_count = by_type.get("code", 0)
    ticket_count = by_type.get("ticket", 0)
    confluence_count = by_type.get("confluence", 0)
    total_docs = stats.get("total_documents", 0)
    disk_mb = stats.get("disk_size_mb", 0.0)

    return Div(id="collection-stats-card", cls="card bg-base-100 shadow border border-base-200 p-5")(
        Div(cls="flex justify-between items-center mb-4")(
            H3(cls="text-base font-bold text-base-content")("Local Knowledge Base Stats"),
            Div(cls="flex gap-2")(
                Button(
                    hx_get="/api/stats",
                    hx_target="#collection-stats-card",
                    hx_swap="outerHTML",
                    cls="uk-button uk-button-default uk-button-xs",
                )("Refresh"),
                Button(
                    hx_post="/api/clear",
                    hx_confirm="Are you sure you want to delete all indexed data in DuckDB?",
                    hx_target="#collection-stats-card",
                    hx_swap="outerHTML",
                    cls="uk-button uk-button-danger uk-button-xs",
                )("Clear DB"),
            ),
        ),
        Div(cls="grid grid-cols-2 md:grid-cols-4 gap-4 text-center")(
            Div(cls="p-3 bg-base-200 rounded-lg")(
                Div(cls="text-2xl font-extrabold text-primary")(str(total_docs)),
                Div(cls="text-xs text-base-content/70")("Total Chunks"),
            ),
            Div(cls="p-3 bg-base-200 rounded-lg")(
                Div(cls="text-2xl font-extrabold text-secondary")(str(code_count)),
                Div(cls="text-xs text-base-content/70")("Code Chunks"),
            ),
            Div(cls="p-3 bg-base-200 rounded-lg")(
                Div(cls="text-2xl font-extrabold text-accent")(str(ticket_count)),
                Div(cls="text-xs text-base-content/70")("ADO Work Items"),
            ),
            Div(cls="p-3 bg-base-200 rounded-lg")(
                Div(cls="text-2xl font-extrabold text-info")(str(confluence_count)),
                Div(cls="text-xs text-base-content/70")("Confluence Wiki"),
            ),
        ),
        Div(cls="mt-3 text-xs text-base-content/60 flex justify-between")(
            Span(f"Disk storage: {disk_mb} MB"),
            Span("Storage engine: DuckDB (Vectors + BM25 FTS)"),
        ),
    )


def IngestionTab(stats: Dict[str, Any]):
    """Tab 1: Ingest Sources view with form, live SSE progress container, and stats."""
    return Div(cls="max-w-5xl mx-auto py-6 px-4 space-y-6")(
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
        error_message and Div(cls="uk-alert uk-alert-danger text-xs p-3 rounded-lg font-mono")(error_message),
        Div(cls="bg-base-300 rounded-lg p-3")(
            Pre(cls="text-xs font-mono text-base-content/80 whitespace-pre-wrap overflow-x-auto max-h-36")(
                log_content
            )
        ),
        status == "completed" and Div(cls="pt-2 flex justify-end")(
            Button(
                hx_get="/api/stats",
                hx_target="#collection-stats-card",
                hx_swap="outerHTML",
                cls="uk-button uk-button-primary uk-button-sm",
            )("Update Stats"),
        ),
    )


def CitationDrawer(documents_with_scores: List[Tuple[Document, float]]):
    """
    MonsterUI Accordion component displaying collapsible citation sources
    used to generate the answer.
    """
    if not documents_with_scores:
        return Div()

    accordion_items = []
    for idx, (doc, score) in enumerate(documents_with_scores, start=1):
        citation = doc.get_citation_tag()
        type_badge_cls = {
            "code": "uk-badge-secondary",
            "ticket": "uk-badge-accent",
            "confluence": "uk-badge-info",
        }.get(doc.doc_type, "uk-badge-default")

        title_component = Div(cls="flex flex-wrap items-center gap-2 text-xs")(
            Span(cls=f"uk-badge {type_badge_cls} font-mono uppercase text-[10px]")(doc.doc_type),
            Span(cls="font-semibold text-base-content")(citation),
            Span(cls="font-mono text-base-content/50 ml-auto")(f"Relevance: {score:.3f}"),
        )

        body_content = Div(cls="p-3 bg-base-200 rounded-md text-xs space-y-2 font-mono")(
            Div(cls="flex justify-between items-center text-base-content/70 text-[11px] pb-1 border-b border-base-300")(
                Span(f"Source URL: "),
                A(href=doc.source_url, target="_blank", rel="noopener noreferrer", cls="text-primary hover:underline truncate max-w-md")(
                    doc.source_url
                ),
            ),
            doc.file_path and Div(cls="text-base-content/70 text-[11px]")(f"File Path: {doc.file_path}"),
            doc.sprint_id and Div(cls="text-base-content/70 text-[11px]")(f"Sprint: {doc.sprint_id}"),
            doc.work_item_id and Div(cls="text-base-content/70 text-[11px]")(f"Work Item ID: {doc.work_item_id}"),
            Pre(cls="bg-base-300 p-3 rounded text-[11px] whitespace-pre-wrap overflow-x-auto max-h-48 border border-base-content/10")(
                doc.content
            ),
        )

        accordion_items.append(
            ui.AccordionItem(
                title_component,
                body_content,
                cls="border border-base-300 rounded-lg p-2 mb-2 bg-base-100",
            )
        )

    return Div(cls="mt-4 pt-3 border-t border-base-300")(
        Div(cls="text-xs font-bold text-base-content/70 uppercase tracking-wider mb-2")(
            f"📚 Retrieved Context Sources ({len(documents_with_scores)})"
        ),
        ui.Accordion(*accordion_items, multiple=True, collapsible=True),
    )


def ChatTab():
    """Tab 2: Ask Chatbot view with message timeline, query input, filter bar, and citations."""
    return Div(cls="max-w-5xl mx-auto py-6 px-4 flex flex-col h-[calc(100vh-140px)]")(
        # Filter row
        Div(cls="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-base-200 mb-4")(
            Div(cls="flex items-center space-x-2")(
                Span(cls="text-xs font-semibold text-base-content/70")("Filter Context:"),
                Select(id="doc-type-filter", name="doc_type_filter", cls="uk-select uk-select-sm text-xs rounded-md w-36")(
                    Option(value="all")("All Sources"),
                    Option(value="code")("Code Only"),
                    Option(value="ticket")("ADO Tickets Only"),
                    Option(value="confluence")("Confluence Only"),
                ),
            ),
            Div(cls="flex items-center space-x-2")(
                Input(
                    id="sprint-filter",
                    name="sprint_filter",
                    type="text",
                    placeholder="Filter by Sprint (e.g. Sprint 42)",
                    cls="uk-input uk-input-sm text-xs rounded-md w-48",
                ),
            ),
        ),

        # Chat history container
        Div(
            id="chat-history",
            cls="flex-1 overflow-y-auto space-y-4 pr-2 pb-4",
        )(
            # Welcome bubble
            Div(cls="flex items-start space-x-3")(
                Div(cls="w-8 h-8 rounded-full bg-primary text-primary-content flex items-center justify-center font-bold text-xs")("AI"),
                Div(cls="flex-1 bg-base-100 border border-base-200 rounded-2xl p-4 shadow-sm")(
                    P(cls="text-sm text-base-content")(
                        "Hello! I am your strictly local, confidential RAG assistant. "
                        "I answer questions using only the codebases, Azure DevOps work items, and Confluence documentation indexed into your local DuckDB database."
                    ),
                    P(cls="text-xs text-base-content/70 mt-2")(
                        "Every statement is grounded in local context with explicit inline citations. "
                        "Try asking questions like: 'How does authentication work in our services?', 'What bugs are open in Sprint 42?', or 'What is our deployment architecture?'"
                    ),
                ),
            ),
        ),

        # Chat input container
        Div(cls="pt-3 border-t border-base-200 mt-2")(
            Form(
                hx_post="/api/chat",
                hx_target="#chat-history",
                hx_swap="beforeend",
                hx_include="#doc-type-filter, #sprint-filter",
                hx_on__after_request="this.reset(); document.getElementById('chat-history').scrollTop = document.getElementById('chat-history').scrollHeight;",
                cls="flex gap-2",
            )(
                Input(
                    id="query-input",
                    name="query",
                    type="text",
                    required=True,
                    autocomplete="off",
                    placeholder="Ask a question about your code, boards, or wiki...",
                    cls="uk-input flex-1 rounded-xl text-sm py-3 px-4",
                ),
                Button(
                    type="submit",
                    cls="uk-button uk-button-primary rounded-xl px-5 flex items-center gap-1",
                )(
                    Span("Ask"),
                    Span(cls="text-xs")("↵"),
                ),
            ),
        ),
    )
