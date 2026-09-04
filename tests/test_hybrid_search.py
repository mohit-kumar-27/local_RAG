"""
Unit tests for hybrid search:
- Query intent classification
- Reciprocal Rank Fusion (RRF) logic
- FlashRank local re-ranking
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from config import EMBED_DIM
from ingestion.base import Document
from rag.hybrid_search import HybridSearcher, infer_query_intent
from rag.reranker import LocalReranker


class TestHybridSearch(unittest.IsolatedAsyncioTestCase):

    def test_query_intent_classification(self):
        # Code intent queries
        self.assertEqual(infer_query_intent("Show me the auth service function implementation"), "code")
        self.assertEqual(infer_query_intent("Where is the user repository class defined?"), "code")

        # Ticket intent queries
        self.assertEqual(infer_query_intent("What is the acceptance criteria for bug #1042?"), "ticket")
        self.assertEqual(infer_query_intent("Show all open tasks in sprint 5"), "ticket")

        # Confluence intent queries
        self.assertEqual(infer_query_intent("What is our architecture policy on secrets in git?"), "confluence")
        self.assertEqual(infer_query_intent("Where can I find the team onboarding guide wiki?"), "confluence")

        # Ambiguous query
        self.assertIsNone(infer_query_intent("What is the system design for payments?"))

    def test_flashrank_reranker(self):
        reranker = LocalReranker(model_name="flashrank")

        doc_auth = Document(
            content="def authenticate(token):\n    # checks OAuth2 bearer tokens and permissions\n    return verify(token)",
            source_url="https://github.com/org/repo/blob/main/auth.py",
            doc_type="code",
        )
        doc_unrelated = Document(
            content="Recipe for banana bread with chocolate chips and walnuts.",
            source_url="https://wiki.corp.com/recipes/banana",
            doc_type="confluence",
        )

        results = reranker.rerank(
            query="OAuth2 authentication verification code",
            documents=[doc_unrelated, doc_auth],
            top_k=2,
        )

        self.assertEqual(len(results), 2)
        top_doc, top_score = results[0]
        # FlashRank should rank the auth code as the top result
        self.assertEqual(top_doc.doc_type, "code")
        self.assertIn("OAuth2", top_doc.content)

    async def test_hybrid_searcher_rrf(self):
        # Mock DuckDB store and Ollama client
        mock_store = MagicMock()
        mock_ollama = MagicMock()
        mock_ollama.embed_single = AsyncMock(return_value=[0.1] * EMBED_DIM)

        doc1 = Document(id="d1", content="doc 1 content", source_url="url1", doc_type="code")
        doc2 = Document(id="d2", content="doc 2 content", source_url="url2", doc_type="ticket")

        # Dense: [doc1, doc2]
        mock_store.search_dense.return_value = [(doc1, 0.9), (doc2, 0.5)]
        # Sparse: [doc2, doc1]
        mock_store.search_sparse.return_value = [(doc2, 10.0), (doc1, 4.0)]

        searcher = HybridSearcher(store=mock_store, ollama_client=mock_ollama)
        results = await searcher.search(query="test search", top_k=2, auto_route=False)

        self.assertEqual(len(results), 2)
        self.assertTrue(any(r[0].id == "d1" for r in results))
        self.assertTrue(any(r[0].id == "d2" for r in results))


if __name__ == "__main__":
    unittest.main()
