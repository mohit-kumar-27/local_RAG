"""
Local reranker module using FlashRank (ONNX CPU, tiny footprint <100MB RAM)
via the rerankers library. Configurable to cross-encoder models with zero code changes.
Strictly local and confidential: no external API calls.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from rerankers import Reranker

from config import BASE_DIR, FLASHRANK_CACHE_DIR, RERANKER_MODEL
from ingestion.base import Document


class LocalReranker:
    """
    Reranks candidate documents using local CPU ONNX models (FlashRank).
    Ensures zero telemetry and zero outbound cloud API calls.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        chosen = model_name or RERANKER_MODEL
        # Disallow generic alias to prevent library default shifts
        if not chosen or chosen.lower() in ("flashrank", "default"):
            chosen = "ms-marco-TinyBERT-L-2-v2"
        self.model_name = chosen
        self.cache_dir = cache_dir or self._resolve_cache_dir()
        self._ranker: Optional[Any] = None
        self._init_attempted: bool = False

    @staticmethod
    def _resolve_cache_dir() -> str:
        """Determines best cache directory across environments."""
        env_dir = os.getenv("FLASHRANK_CACHE_DIR")
        if env_dir:
            return env_dir
        if FLASHRANK_CACHE_DIR.exists():
            return str(FLASHRANK_CACHE_DIR.resolve())
        local_cache = Path("./.flashrank_cache")
        if local_cache.exists():
            return str(local_cache.resolve())
        user_cache = Path.home() / ".cache" / "flashrank"
        user_cache.mkdir(parents=True, exist_ok=True)
        return str(user_cache)

    def _get_ranker(self):
        if self._ranker is not None:
            return self._ranker
        if self._init_attempted:
            return None

        self._init_attempted = True
        # Initialize rerankers.Reranker with explicit model name and designated cache directory
        try:
            self._ranker = Reranker(
                model_name=self.model_name,
                model_type="flashrank",
                verbose=0,
                cache_dir=self.cache_dir,
            )
            return self._ranker
        except Exception:
            pass

        # Fallback directly to native flashrank ranker with explicit model name and cache directory
        try:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
            return self._ranker
        except Exception as e:
            print(f"Warning: Reranker initialization failed for model '{self.model_name}': {e}. Falling back to initial ranking.")
            self._ranker = None
            return None

    def prewarm(self) -> bool:
        """Pre-warms the reranker at startup so chat queries don't stall on first run."""
        try:
            ranker = self._get_ranker()
            return ranker is not None
        except Exception:
            return False

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
