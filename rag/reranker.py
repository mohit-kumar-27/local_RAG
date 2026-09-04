"""
Local reranker module using FlashRank (ONNX CPU, tiny footprint <100MB RAM)
via the rerankers library. Configurable to cross-encoder models with zero code changes.
Strictly local and confidential: no external API calls.
"""

from typing import Any, List, Optional, Tuple

from rerankers import Reranker

from config import RERANKER_MODEL
from ingestion.base import Document


class LocalReranker:
    """
    Reranks candidate documents using local CPU ONNX models (FlashRank).
    Ensures zero telemetry and zero outbound cloud API calls.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._ranker: Optional[Any] = None

    def _get_ranker(self):
        if self._ranker is None:
            # Initialize local ranker (default: flashrank)
            try:
                self._ranker = Reranker(self.model_name, verbose=0)
            except Exception as e:
                # Fallback directly to flashrank ranker
                try:
                    from flashrank import Ranker
                    self._ranker = Ranker()
                except Exception:
                    print(f"Warning: Reranker initialization failed: {e}")
                    self._ranker = None
        return self._ranker

    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Reranks a list of candidate Documents against the user query.
        Returns the top_k (Document, score) pairs.
        """
        if not documents:
            return []

        if len(documents) <= 1:
            return [(documents[0], 1.0)]

        ranker = self._get_ranker()
        if ranker is None:
            # Fallback: preserve original order
            return [(d, 1.0 / (i + 1)) for i, d in enumerate(documents[:top_k])]

        # Extract text snippets for ranking
        doc_texts = [d.content for d in documents]

        try:
            # Check if it is a rerankers.Reranker instance
            if hasattr(ranker, "rank"):
                ranked_result = ranker.rank(query=query, docs=doc_texts)
                # ranked_result.results contains items with doc_id (index in original list) and score
                reranked_docs: List[Tuple[Document, float]] = []
                for res in ranked_result.results[:top_k]:
                    orig_doc = documents[res.doc_id]
                    reranked_docs.append((orig_doc, float(res.score)))
                return reranked_docs
            elif hasattr(ranker, "rerank"):
                # FlashRank native interface
                from flashrank import RerankRequest
                passages = [{"id": i, "text": d.content} for i, d in enumerate(documents)]
                req = RerankRequest(query=query, passages=passages)
                results = ranker.rerank(req)
                reranked_docs = []
                for res in results[:top_k]:
                    orig_doc = documents[res["id"]]
                    reranked_docs.append((orig_doc, float(res["score"])))
                return reranked_docs
        except Exception as e:
            print(f"Reranking encountered an error: {e}. Falling back to initial ranking.")
            return [(d, 1.0 / (i + 1)) for i, d in enumerate(documents[:top_k])]

        return [(d, 1.0 / (i + 1)) for i, d in enumerate(documents[:top_k])]
