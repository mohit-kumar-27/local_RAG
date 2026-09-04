"""
RAG core module: chunking, DuckDB store, hybrid search, Ollama client, reranker, and RAG pipeline.
"""

from .chunking import CodeChunker, MarkdownSectionChunker, count_tokens
from .duckdb_store import DuckDBStore
from .ollama_client import OllamaClient
from .reranker import LocalReranker
from .hybrid_search import HybridSearcher, infer_query_intent
from .rag_pipeline import RAGPipeline, SYSTEM_PROMPT, format_context_passages

__all__ = [
    "CodeChunker",
    "MarkdownSectionChunker",
    "count_tokens",
    "DuckDBStore",
    "OllamaClient",
    "LocalReranker",
    "HybridSearcher",
    "infer_query_intent",
    "RAGPipeline",
    "SYSTEM_PROMPT",
    "format_context_passages",
]
