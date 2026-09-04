"""
Hybrid search module combining Dense Vector Search and Sparse BM25 Keyword Search
via Reciprocal Rank Fusion (RRF) and FlashRank re-ranking.
Includes query intent routing for metadata filtering.
"""

import re
from typing import Dict, List, Optional, Tuple

from config import RRF_K, TOP_K_FINAL, TOP_K_HYBRID
from ingestion.base import Document
from rag.duckdb_store import DuckDBStore
from rag.ollama_client import OllamaClient
from rag.reranker import LocalReranker


def infer_query_intent(query: str) -> Optional[str]:
    """
    Infers document type filtering based on natural language query keywords.
    Returns: 'code', 'ticket', 'confluence', or None (search across all).
    """
    q_lower = query.lower()

    # Code intent
    code_keywords = [
        r"\bcode\b", r"\bfunction\b", r"\bmethod\b", r"\bclass\b",
        r"\bfile\b", r"\bsyntax\b", r"\bimplementation\b", r"\bapi\b",
        r"\bdef\b", r"\brepository\b", r"\brepo\b", r"\.py\b", r"\.ts\b",
    ]
    if any(re.search(pat, q_lower) for pat in code_keywords):
        return "code"

    # Ticket / Board intent
    ticket_keywords = [
        r"\bbug\b", r"\bissue\b", r"\bticket\b", r"\btask\b",
        r"\buser story\b", r"\bepic\b", r"\bsprint\b", r"\bacceptance criteria\b",
        r"\bado\b", r"\bboard\b", r"\bwork item\b",
    ]
    if any(re.search(pat, q_lower) for pat in ticket_keywords):
        return "ticket"

    # Confluence / Documentation intent
    docs_keywords = [
        r"\bconfluence\b", r"\bwiki\b", r"\bdocument\b", r"\bdocumentation\b",
        r"\bguide\b", r"\bonboarding\b", r"\barchitecture\b", r"\bdesign doc\b",
        r"\brfc\b", r"\bpolicy\b",
    ]
    if any(re.search(pat, q_lower) for pat in docs_keywords):
        return "confluence"

    return None


class HybridSearcher:
    """
    Orchestrates dense + sparse retrieval, Reciprocal Rank Fusion,
    intent filtering, and FlashRank re-ranking.
    """

    def __init__(
        self,
        store: DuckDBStore,
        ollama_client: OllamaClient,
        reranker: Optional[LocalReranker] = None,
        rrf_k: int = RRF_K,
    ):
        self.store = store
        self.ollama = ollama_client
        self.reranker = reranker or LocalReranker()
        self.rrf_k = rrf_k

    async def search(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        top_k_candidates: int = TOP_K_HYBRID,
        doc_type_filter: Optional[str] = None,
        sprint_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        auto_route: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Executes hybrid retrieval:
        1. Query intent routing (if doc_type_filter is not manually specified)
        2. Embedding generation via local Ollama
        3. Parallel dense (cosine) & sparse (BM25) queries in DuckDB
        4. Reciprocal Rank Fusion (RRF)
        5. FlashRank re-ranking on CPU
        """
        # Determine effective document type filter
        effective_filter = doc_type_filter
        if (not effective_filter or effective_filter.lower() == "all") and auto_route:
            inferred = infer_query_intent(query)
            # Only set if confident, otherwise search all
            if inferred:
                effective_filter = inferred

        # 1. Compute query vector
        query_vector = await self.ollama.embed_single(query)

        # 2. Retrieve dense candidates
        dense_results = self.store.search_dense(
            query_vector=query_vector,
            top_k=top_k_candidates,
            doc_type=effective_filter,
            sprint_id=sprint_filter,
            source_url_filter=source_filter,
        )

        # 3. Retrieve sparse candidates
        sparse_results = self.store.search_sparse(
            query_text=query,
            top_k=top_k_candidates,
            doc_type=effective_filter,
            sprint_id=sprint_filter,
            source_url_filter=source_filter,
        )

        # 4. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Process dense rankings
        for rank, (doc, _) in enumerate(dense_results, start=1):
            doc_map[doc.id] = doc
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Process sparse rankings
        for rank, (doc, _) in enumerate(sparse_results, start=1):
            doc_map[doc.id] = doc
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort fused candidates by RRF score
        fused_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_docs = [doc_map[doc_id] for doc_id, _ in fused_sorted[:top_k_candidates]]

        if not candidate_docs:
            return []

        # 5. Local FlashRank Re-ranking
        reranked = self.reranker.rerank(query=query, documents=candidate_docs, top_k=top_k)
        return reranked
