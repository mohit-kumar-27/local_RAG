"""
Command-Line Interface (CLI) for the Local RAG System.
Supports headless ingestion, semantic/hybrid querying with citations,
collection statistics, and local Ollama health checks.
"""

import argparse
import asyncio
import sys
from typing import Optional

import config
from ingestion.ado_loader import AdoBoardLoader, AdoRepoLoader
from ingestion.confluence_loader import ConfluenceLoader
from ingestion.github_loader import GithubCodeLoader
from rag.duckdb_store import DuckDBStore
from rag.hybrid_search import HybridSearcher
from rag.ollama_client import OllamaClient
from rag.rag_pipeline import RAGPipeline
from rag.reranker import LocalReranker


async def check_health(ollama: OllamaClient):
    print("\n--- Local System Health & Model Verification ---")
    ok, models, msg = await ollama.check_connection()
    if ok:
        print(f"[OK] Ollama is active on {ollama.base_url}")
        print(f"     Installed models: {', '.join(models) if models else 'None'}")
        
        embed_ok = any(ollama.embed_model in m for m in models)
        llm_ok = any(ollama.llm_model in m for m in models)
        
        print(f"     Embed model ({ollama.embed_model}): {'Available' if embed_ok else 'MISSING (Run: ollama pull ' + ollama.embed_model + ')'}")
        print(f"     Active LLM ({ollama.llm_model}): {'Available' if llm_ok else 'MISSING (Run: ollama pull ' + ollama.llm_model + ')'}")
    else:
        print(f"[FAIL] Ollama connection failed: {msg}")
        print("       Please ensure Ollama is running: 'ollama serve' or open the Ollama app.")


def print_stats(store: DuckDBStore):
    stats = store.get_collection_stats()
    print("\n--- DuckDB Local Knowledge Base Statistics ---")
    print(f"Total Chunks:     {stats['total_documents']}")
    print(f"Disk Size:        {stats['disk_size_mb']} MB")
    print(f"Unique Sources:   {stats['unique_sources']}")
    print("Breakdown by Type:")
    for doc_type, count in stats.get("by_type", {}).items():
        print(f"  - {doc_type:12}: {count} chunks")
    print("----------------------------------------------")


async def run_ingestion(
    source_type: str,
    url: str,
    branch: Optional[str] = None,
    token: Optional[str] = None,
):
    print(f"\n[Ingest] Starting ingestion for {source_type.upper()}: {url}")
    store = DuckDBStore()
    ollama = OllamaClient()

    if source_type == "github":
        loader = GithubCodeLoader(repo_url_or_slug=url, pat_token=token, branch=branch)
    elif source_type == "ado_board":
        loader = AdoBoardLoader(board_url_or_query=url, pat_token=token)
    elif source_type == "ado_repo":
        loader = AdoRepoLoader(repo_url=url, pat_token=token, branch=branch or "main")
    elif source_type == "confluence":
        loader = ConfluenceLoader(url_or_space=url, api_token=token)
    else:
        print(f"[Error] Unsupported source type: {source_type}")
        return

    print("[1/4] Extracting and chunking documents from source...")
    documents = await loader.load()
    print(f"      Extracted {len(documents)} raw document chunks.")

    if not documents:
        print("[Done] No documents extracted.")
        return

    print("[2/4] Checking content hashes for deduplication...")
    all_hashes = [d.content_hash for d in documents]
    existing = store.filter_existing_hashes(all_hashes)
    new_docs = [d for d in documents if d.content_hash not in existing]
    print(f"      {len(existing)} unchanged chunks skipped. {len(new_docs)} new chunks require embedding.")

    if not new_docs:
        print("[Done] All documents are already indexed and up to date.")
        return

    print(f"[3/4] Generating vector embeddings via Ollama ({ollama.embed_model})...")
    batch_size = max(1, config.EMBEDDING_BATCH_SIZE)
    for i in range(0, len(new_docs), batch_size):
        batch = new_docs[i : i + batch_size]
        batch_embeddings = await ollama.embed_texts([d.content for d in batch])
        for doc, emb in zip(batch, batch_embeddings):
            doc.embedding = emb
        sys.stdout.write(f"\r      Progress: {min(i + batch_size, len(new_docs))}/{len(new_docs)} chunks embedded")
        sys.stdout.flush()
    print()

    print("[4/4] Writing to DuckDB and updating BM25 index...")
    inserted = store.insert_documents(new_docs)
    print(f"[Done] Successfully indexed {inserted} new chunks into DuckDB.")
    print_stats(store)


async def run_query(
    question: str,
    doc_filter: Optional[str] = None,
    sprint: Optional[str] = None,
):
    print(f"\n[Query] \"{question}\"")
    store = DuckDBStore()
    ollama = OllamaClient()
    reranker = LocalReranker()
    searcher = HybridSearcher(store=store, ollama_client=ollama, reranker=reranker)
    pipeline = RAGPipeline(hybrid_searcher=searcher, ollama_client=ollama)

    print("[Search] Executing Hybrid Search (Dense Cosine + Sparse BM25 + FlashRank)...")
    retrieved_docs, stream_gen = await pipeline.answer_stream(
        query=question,
        doc_type_filter=doc_filter,
        sprint_filter=sprint,
    )

    print(f"\n--- Retrieved Context Sources ({len(retrieved_docs)}) ---")
    for idx, (doc, score) in enumerate(retrieved_docs, start=1):
        print(f"[{idx}] {doc.get_citation_tag()} (Score: {score:.3f})")
        print(f"    URL: {doc.source_url}")

    print("\n--- Grounded Answer (Streaming) ---")
    async for token in stream_gen:
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n-----------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Confidential Local RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # health
    subparsers.add_parser("health", help="Check local Ollama server and installed models")

    # stats
    subparsers.add_parser("stats", help="Show DuckDB indexed document statistics")

    # clear
    subparsers.add_parser("clear", help="Clear all documents from DuckDB")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a repository, board, or wiki")
    ingest_parser.add_argument("--type", required=True, choices=["github", "ado_board", "ado_repo", "confluence"])
    ingest_parser.add_argument("--url", required=True, help="URL or identifier")
    ingest_parser.add_argument("--branch", help="Git branch to clone")
    ingest_parser.add_argument("--token", help="PAT or API token override")

    # query
    query_parser = subparsers.add_parser("query", help="Ask a question against local knowledge base")
    query_parser.add_argument("question", help="Natural language question")
    query_parser.add_argument("--filter", choices=["code", "ticket", "confluence"], help="Metadata type filter")
    query_parser.add_argument("--sprint", help="Sprint filter")

    args = parser.parse_args()

    if args.command == "health":
        asyncio.run(check_health(OllamaClient()))
    elif args.command == "stats":
        print_stats(DuckDBStore())
    elif args.command == "clear":
        DuckDBStore().clear_all()
        print("DuckDB database cleared.")
        print_stats(DuckDBStore())
    elif args.command == "ingest":
        asyncio.run(run_ingestion(args.type, args.url, branch=args.branch, token=args.token))
    elif args.command == "query":
        asyncio.run(run_query(args.question, doc_filter=args.filter, sprint=args.sprint))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
