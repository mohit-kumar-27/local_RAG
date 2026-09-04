"""
Unit and integration tests for incremental indexing:
- File-level incremental sync (add, modify, delete)
- Content hash deduplication
- Purging stale chunks upon modification or deletion
"""

import tempfile
import unittest
from pathlib import Path

from config import EMBED_DIM
from ingestion.base import Document
from rag.duckdb_store import DuckDBStore


class TestIncrementalIndexing(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_incremental.duckdb"
        self.store = DuckDBStore(db_path=self.db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_incremental_file_modification(self):
        # 1. Index initial version of a file
        vec1 = [0.1] * EMBED_DIM
        doc_v1 = Document(
            content="def calculate_vat(price): return price * 0.20",
            source_url="https://github.com/org/repo/blob/main/vat.py",
            doc_type="code",
            file_path="vat.py",
            embedding=vec1,
        )
        self.store.insert_documents([doc_v1])

        # Verify it exists
        stats = self.store.get_collection_stats()
        self.assertEqual(stats["total_documents"], 1)
        results = self.store.search_sparse("calculate_vat", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertIn("0.20", results[0][0].content)

        # 2. File is modified: VAT changes to 0.21
        # In incremental mode, the loader purges old chunks for that file
        self.store.delete_documents_by_file("vat.py", source_prefix="org/repo")

        # Insert new version
        vec2 = [0.2] * EMBED_DIM
        doc_v2 = Document(
            content="def calculate_vat(price): return price * 0.21",
            source_url="https://github.com/org/repo/blob/main/vat.py",
            doc_type="code",
            file_path="vat.py",
            embedding=vec2,
        )
        self.store.insert_documents([doc_v2])

        # Verify only the updated version remains (no duplicate/stale version!)
        stats = self.store.get_collection_stats()
        self.assertEqual(stats["total_documents"], 1)
        results = self.store.search_sparse("calculate_vat", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertIn("0.21", results[0][0].content)
        self.assertNotIn("0.20", results[0][0].content)

    def test_incremental_file_deletion(self):
        # Index two files
        vec = [0.1] * EMBED_DIM
        doc_a = Document(
            content="def active_service(): pass",
            source_url="https://github.com/org/repo/blob/main/service.py",
            doc_type="code",
            file_path="service.py",
            embedding=vec,
        )
        doc_b = Document(
            content="def deprecated_service(): pass",
            source_url="https://github.com/org/repo/blob/main/deprecated.py",
            doc_type="code",
            file_path="deprecated.py",
            embedding=vec,
        )
        self.store.insert_documents([doc_a, doc_b])
        self.assertEqual(self.store.get_collection_stats()["total_documents"], 2)

        # File deprecated.py is deleted in git
        self.store.delete_documents_by_file("deprecated.py")

        # Verify only service.py remains
        self.assertEqual(self.store.get_collection_stats()["total_documents"], 1)
        results = self.store.search_sparse("deprecated", top_k=5)
        self.assertEqual(len(results), 0)

    def test_incremental_work_item_update(self):
        # Work Item initially Active
        vec = [0.1] * EMBED_DIM
        item_v1 = Document(
            content="# Bug #1042: Null pointer exception\nState: Active\nAssignee: Alice",
            source_url="https://dev.azure.com/org/proj/_workitems/edit/1042",
            doc_type="ticket",
            work_item_id="1042",
            embedding=vec,
        )
        self.store.insert_documents([item_v1])
        self.assertEqual(self.store.get_collection_stats()["total_documents"], 1)

        # Ticket is updated in ADO to Resolved with new comment
        self.store.delete_documents_by_work_item("1042")
        item_v2 = Document(
            content="# Bug #1042: Null pointer exception\nState: Resolved\nAssignee: Alice\nFixed in commit abcd123",
            source_url="https://dev.azure.com/org/proj/_workitems/edit/1042",
            doc_type="ticket",
            work_item_id="1042",
            embedding=vec,
        )
        self.store.insert_documents([item_v2])

        # Verify single updated record
        self.assertEqual(self.store.get_collection_stats()["total_documents"], 1)
        results = self.store.search_sparse("Null pointer exception", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertIn("State: Resolved", results[0][0].content)


if __name__ == "__main__":
    unittest.main()
