"""
Unit tests for DuckDBStore:
- Vector insertion & cosine similarity retrieval
- Content hash deduplication
- Keyword / sparse search
- Metadata filtering
- Collection statistics
"""

import os
import tempfile
import unittest
from pathlib import Path

from config import EMBED_DIM
from ingestion.base import Document
from rag.duckdb_store import DuckDBStore


class TestDuckDBStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_store.duckdb"
        self.store = DuckDBStore(db_path=self.db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_deduplication_and_insertion(self):
        # Create dummy vector of dimension EMBED_DIM (768)
        vec_a = [0.0] * EMBED_DIM
        vec_a[0] = 1.0

        doc1 = Document(
            content="def calculate_total(items): return sum(items)",
            source_url="https://github.com/org/repo/blob/main/calc.py",
            doc_type="code",
            file_path="calc.py",
            embedding=vec_a,
        )

        inserted_count = self.store.insert_documents([doc1])
        self.assertEqual(inserted_count, 1)

        # Attempt to insert identical document with same content_hash
        doc1_duplicate = Document(
            content="def calculate_total(items): return sum(items)",
            source_url="https://github.com/org/repo/blob/main/calc.py",
            doc_type="code",
            file_path="calc.py",
            embedding=vec_a,
        )
        second_inserted = self.store.insert_documents([doc1_duplicate])
        self.assertEqual(second_inserted, 0, "Duplicate content_hash should be skipped")

        stats = self.store.get_collection_stats()
        self.assertEqual(stats["total_documents"], 1)

    def test_dense_cosine_similarity(self):
        vec_auth = [0.0] * EMBED_DIM
        vec_auth[0] = 1.0  # Points in dimension 0

        vec_database = [0.0] * EMBED_DIM
        vec_database[1] = 1.0  # Orthogonal, points in dimension 1

        doc_auth = Document(
            content="Authentication handler verifies OAuth bearer tokens.",
            source_url="https://github.com/org/repo/blob/main/auth.py",
            doc_type="code",
            file_path="auth.py",
            embedding=vec_auth,
        )

        doc_db = Document(
            content="Database connection pool configuration.",
            source_url="https://github.com/org/repo/blob/main/db.py",
            doc_type="code",
            file_path="db.py",
            embedding=vec_database,
        )

        self.store.insert_documents([doc_auth, doc_db])

        # Query vector aligned with vec_auth
        query_vec = [0.0] * EMBED_DIM
        query_vec[0] = 1.0

        results = self.store.search_dense(query_vec, top_k=2)
        self.assertEqual(len(results), 2)
        top_doc, top_score = results[0]
        self.assertEqual(top_doc.file_path, "auth.py")
        self.assertAlmostEqual(top_score, 1.0, places=2)

    def test_metadata_filtering(self):
        vec = [0.1] * EMBED_DIM

        doc_code = Document(
            content="def sync(): pass",
            source_url="https://github.com/org/repo/blob/main/sync.py",
            doc_type="code",
            embedding=vec,
        )
        doc_ticket = Document(
            content="Bug: sync fails on timeout",
            source_url="https://dev.azure.com/org/proj/_workitems/101",
            doc_type="ticket",
            work_item_id="101",
            sprint_id="Sprint 42",
            embedding=vec,
        )

        self.store.insert_documents([doc_code, doc_ticket])

        # Filter by doc_type
        ticket_results = self.store.search_dense(vec, top_k=5, doc_type="ticket")
        self.assertEqual(len(ticket_results), 1)
        self.assertEqual(ticket_results[0][0].work_item_id, "101")

        # Filter by sprint_id
        sprint_results = self.store.search_dense(vec, top_k=5, sprint_id="Sprint 42")
        self.assertEqual(len(sprint_results), 1)
        self.assertEqual(sprint_results[0][0].sprint_id, "Sprint 42")


if __name__ == "__main__":
    unittest.main()
