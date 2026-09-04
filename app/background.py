"""
Background ingestion job manager with Starlette background task support
and real-time SSE progress event streaming.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid

from config import EMBEDDING_BATCH_SIZE
from ingestion.base import BaseLoader, Document
from rag.duckdb_store import DuckDBStore
from rag.ollama_client import OllamaClient


@dataclass
class IngestionJob:
    """State of an asynchronous data ingestion operation."""
    id: str
    source_type: str
    url: str
    status: str = "queued"  # 'queued', 'running', 'completed', 'failed'
    stage: str = "Initialized"
    progress: int = 0
    total_found: int = 0
    new_indexed: int = 0
    skipped_duplicates: int = 0
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {message}")


# Active jobs cache
JOBS: Dict[str, IngestionJob] = {}


def create_job(source_type: str, url: str) -> IngestionJob:
    job_id = str(uuid.uuid4())[:8]
    job = IngestionJob(id=job_id, source_type=source_type, url=url)
    JOBS[job_id] = job
    return job


def get_job(job_id: str) -> Optional[IngestionJob]:
    return JOBS.get(job_id)


async def execute_ingestion(
    job_id: str,
    loader: BaseLoader,
    store: DuckDBStore,
    ollama: OllamaClient,
):
    """
    Executes the ingestion workflow in the background:
    1. Extract and chunk documents from the source (GitHub, ADO, Confluence).
    2. Check content_hash deduplication against DuckDB.
    3. Generate 768-dim embeddings in modest batches via local Ollama.
    4. Persist newly embedded documents and rebuild BM25 index.
    """
    job = get_job(job_id)
    if not job:
        return

    job.status = "running"
    job.stage = "Connecting to source..."
    job.progress = 10
    job.add_log(f"Starting ingestion for {job.source_type}: {job.url}")

    try:
        # Step 1: Load and chunk
        job.stage = "Cloning / Fetching and chunking documents..."
        job.progress = 25
        job.add_log("Extracting content and applying structure-aware chunking...")
        documents = await loader.load()
        job.total_found = len(documents)
        job.add_log(f"Extraction finished. Extracted {len(documents)} document chunks.")

        if not documents:
            job.stage = "Complete (No documents found)"
            job.status = "completed"
            job.progress = 100
            job.add_log("No documents matching filter criteria were found.")
            return

        # Step 2: Deduplication check
        job.stage = "Checking content hashes for deduplication..."
        job.progress = 40
        all_hashes = [d.content_hash for d in documents]
        existing_hashes = store.filter_existing_hashes(all_hashes)

        docs_to_embed = [d for d in documents if d.content_hash not in existing_hashes]
        job.skipped_duplicates = len(documents) - len(docs_to_embed)
        job.add_log(
            f"Deduplication complete: {job.skipped_duplicates} unchanged chunks skipped, "
            f"{len(docs_to_embed)} new or modified chunks require embedding."
        )

        if not docs_to_embed:
            job.stage = "Complete (All documents up to date)"
            job.status = "completed"
            job.progress = 100
            job.add_log("All documents are already indexed and up-to-date in DuckDB.")
            return

        # Step 3: Embed in modest batches
        job.stage = "Generating vector embeddings via local Ollama (nomic-embed-text)..."
        batch_size = max(1, EMBEDDING_BATCH_SIZE)
        total_batches = (len(docs_to_embed) + batch_size - 1) // batch_size
        texts_to_embed = [d.content for d in docs_to_embed]

        for b_idx in range(total_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, len(docs_to_embed))
            batch_texts = texts_to_embed[start:end]

            batch_embeddings = await ollama.embed_texts(batch_texts)
            for d, emb in zip(docs_to_embed[start:end], batch_embeddings):
                d.embedding = emb

            # Update progress between 40% and 85%
            pct = 40 + int(45 * ((b_idx + 1) / total_batches))
            job.progress = pct
            job.stage = f"Embedding chunks: {end}/{len(docs_to_embed)} (batch {b_idx + 1}/{total_batches})..."

        job.add_log(f"Generated {len(docs_to_embed)} vector embeddings.")

        # Step 4: Insert into DuckDB
        job.stage = "Writing to DuckDB and rebuilding FTS index..."
        job.progress = 90
        inserted_count = store.insert_documents(docs_to_embed)
        job.new_indexed = inserted_count

        job.progress = 100
        job.stage = "Complete"
        job.status = "completed"
        job.add_log(f"Successfully indexed {inserted_count} chunks into DuckDB.")

    except Exception as e:
        job.status = "failed"
        job.stage = "Failed"
        job.error_message = str(e)
        job.add_log(f"ERROR: {str(e)}")
        job.add_log(traceback.format_exc())
