"""
FastHTML single ASGI application.
Combines Starlette backend routes, MonsterUI component rendering,
and real-time SSE token/progress streaming into a single process.
"""

import asyncio
import os
import urllib.parse
import uuid
from typing import Optional

from fasthtml.common import (
    Body, Div, FastHTML, Html, NotStr, P, Script, Span, fast_app, sse_message
)
import mistletoe
import monsterui.all as ui
from starlette.responses import StreamingResponse

import config
from app.background import create_job, execute_ingestion, get_job
from app.ui_components import (
    AppHeader, ChatTab, CitationDrawer, CollectionStatsCard,
    IngestProgressSSEComponent, IngestProgressUpdateCard, IngestionTab, TabNavigation
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
        # HTMX SSE Extension
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.2/sse.js"),
        # Tailwind typography, auto-scroll & SSE close handling
        Script("""
        document.addEventListener('htmx:afterSwap', function(evt) {
            const chatBox = document.getElementById('chat-history');
            if (chatBox) {
                chatBox.scrollTop = chatBox.scrollHeight;
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


def MainLayout(active_tab: str = "chat", ollama_ok: bool = True):
    """Main application shell with fixed full-viewport layout."""
    stats = store.get_collection_stats()
    content = IngestionTab(stats) if active_tab == "ingest" else ChatTab()

    return Div(id="main-content", cls="h-screen bg-base-200 text-base-content flex flex-col font-sans overflow-hidden")(
        AppHeader(ollama_connected=ollama_ok, current_model=config.get_active_llm_model()),
        TabNavigation(active_tab=active_tab),
        Div(id="tab-content", cls="flex-1 flex flex-col min-h-0 overflow-hidden")(
            content
        ),
    )


@rt("/")
async def get(tab: Optional[str] = "chat"):
    """Main route serving Ingest and Chatbot tabs."""
    ok, _, _ = await ollama.check_connection()
    return MainLayout(active_tab=tab or "chat", ollama_ok=ok)


@rt("/tab/{tab_name}")
async def get_tab(tab_name: str):
    """Swaps tab content via HTMX."""
    if tab_name == "ingest":
        stats = store.get_collection_stats()
        return IngestionTab(stats)
    return ChatTab()


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
async def post_toggle_ram():
    """Toggles LOW_RAM_MODE between 8B and 3B models."""
    config.LOW_RAM_MODE = not config.LOW_RAM_MODE
    ok, _, _ = await ollama.check_connection()
    return MainLayout(active_tab="chat", ollama_ok=ok)


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


@rt("/api/chat")
async def post_chat(
    query: str,
    doc_type_filter: Optional[str] = "all",
    sprint_filter: Optional[str] = None,
):
    """
    Handles user chat submission:
    Returns user chat bubble + SSE stream container for active assistant response.
    """
    stream_id = str(uuid.uuid4())[:8]
    encoded_query = urllib.parse.quote(query.strip())
    encoded_doc_filter = urllib.parse.quote(doc_type_filter or "all")
    encoded_sprint = urllib.parse.quote(sprint_filter.strip() if sprint_filter else "")

    # User message element
    user_bubble = Div(cls="flex items-start justify-end space-x-3 w-full")(
        Div(cls="bg-primary text-primary-content rounded-2xl p-4 shadow-sm max-w-2xl text-sm leading-relaxed whitespace-pre-wrap")(
            P(query.strip())
        ),
        Div(cls="w-9 h-9 rounded-full bg-base-300 text-base-content flex-shrink-0 flex items-center justify-center font-bold text-xs shadow-sm")("YOU"),
    )

    # Assistant SSE streaming placeholder element
    stream_url = f"/api/chat/stream/{stream_id}?q={encoded_query}&doc_type={encoded_doc_filter}&sprint={encoded_sprint}"
    assistant_placeholder = Div(
        id=f"stream-container-{stream_id}",
        hx_ext="sse",
        sse_connect=stream_url,
        sse_swap="message",
        sse_close="close",
        hx_target=f"#response-box-{stream_id}",
        cls="flex items-start space-x-3 w-full",
    )(
        Div(cls="w-9 h-9 rounded-full bg-secondary text-secondary-content flex-shrink-0 flex items-center justify-center font-bold text-xs shadow-sm")("AI"),
        Div(
            id=f"response-box-{stream_id}",
            cls="flex-1 bg-base-100 border border-base-300 rounded-2xl p-5 shadow-sm min-w-0 space-y-3",
        )(
            Div(cls="flex items-center space-x-2 text-xs text-base-content/70")(
                Div(cls="animate-spin h-4 w-4 border-2 border-secondary border-t-transparent rounded-full"),
                Span("Searching local knowledge base & generating grounded response..."),
            )
        ),
    )

    return Div(user_bubble, assistant_placeholder)


@rt("/api/chat/stream/{stream_id}")
async def get_chat_stream(
    stream_id: str,
    q: str,
    doc_type: Optional[str] = "all",
    sprint: Optional[str] = None,
):
    """
    SSE endpoint:
    Streams token-by-token generation from Ollama, followed by the collapsible Citation Drawer.
    Explicitly emits 'close' event upon completion to terminate EventSource cleanly.
    """
    query = urllib.parse.unquote(q)
    doc_filter = urllib.parse.unquote(doc_type) if doc_type != "all" else None
    sprint_f = urllib.parse.unquote(sprint) if sprint else None

    async def event_generator():
        try:
            # 1. Start RAG retrieval and stream
            retrieved_docs, token_stream = await rag_pipeline.answer_stream(
                query=query,
                doc_type_filter=doc_filter,
                sprint_filter=sprint_f,
            )

            accumulated_response = ""
            async for token in token_stream:
                accumulated_response += token

                # Parse markdown to HTML
                html_body = mistletoe.markdown(accumulated_response)

                streaming_element = Div(
                    Div(cls="prose prose-sm max-w-none text-base-content leading-relaxed")(
                        NotStr(html_body),
                        Span(cls="inline-block w-2 h-4 ml-1 bg-secondary animate-pulse align-middle"),
                    )
                )
                yield sse_message(streaming_element)

            # 2. Final message with collapsible citation drawer
            final_html_body = mistletoe.markdown(accumulated_response)
            citations_component = CitationDrawer(retrieved_docs)

            final_element = Div(
                Div(cls="prose prose-sm max-w-none text-base-content leading-relaxed")(
                    NotStr(final_html_body)
                ),
                citations_component,
            )
            yield sse_message(final_element)
            # Send close event to prevent browser EventSource from reconnecting and looping
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
