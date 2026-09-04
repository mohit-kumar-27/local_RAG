"""
Unit tests for chunking module:
- AST-based Python chunking
- Generic multi-language structural chunking
- Hierarchy-preserving Markdown section chunking
- Content hash determinism
"""

import unittest
from ingestion.base import compute_content_hash, Document
from rag.chunking import CodeChunker, MarkdownSectionChunker, count_tokens


class TestChunking(unittest.TestCase):

    def setUp(self):
        self.code_chunker = CodeChunker(target_tokens=100, max_tokens=250)
        self.md_chunker = MarkdownSectionChunker(target_tokens=100, max_tokens=250)

    def test_content_hash_determinism(self):
        text1 = "def authenticate():\n    return True\n"
        text2 = "def authenticate():\n    return True   "
        text3 = "def authenticate():\n    return False"
        # Trailing spaces stripped should yield same hash
        self.assertEqual(compute_content_hash(text1), compute_content_hash(text2))
        self.assertNotEqual(compute_content_hash(text1), compute_content_hash(text3))

    def test_ast_python_chunking(self):
        sample_py = '''"""Module docstring for auth service."""

def verify_token(token: str) -> bool:
    """Verifies token validity."""
    if not token:
        return False
    return token.startswith("Bearer ")

class UserManager:
    """Manages user sessions."""
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: str):
        return self.db.find(user_id)
'''
        docs = self.code_chunker.chunk_python_file(
            sample_py, file_path="auth/service.py", source_url="https://github.com/org/repo/blob/main/auth/service.py"
        )
        self.assertGreaterEqual(len(docs), 1)
        # Check that metadata has symbol information
        symbols = [d.metadata.get("symbol_name") for d in docs]
        self.assertTrue("verify_token" in symbols or "UserManager" in symbols)
        for doc in docs:
            self.assertEqual(doc.doc_type, "code")
            self.assertIn("service.py", doc.content)
            self.assertTrue(doc.get_citation_tag().startswith("[Source: auth/service.py"))

    def test_generic_code_chunking(self):
        sample_ts = "\n".join([f"export const item_{i} = () => {{ console.log({i}); }};" for i in range(40)])
        docs = self.code_chunker.chunk_generic_code(
            sample_ts, file_path="src/utils.ts", source_url="https://github.com/org/repo/blob/main/src/utils.ts"
        )
        self.assertGreater(len(docs), 1)
        for doc in docs:
            self.assertIn("# File: src/utils.ts", doc.content)
            self.assertIn("start_line", doc.metadata)
            self.assertIn("end_line", doc.metadata)

    def test_markdown_section_hierarchy(self):
        sample_md = """# Engineering Architecture
Welcome to our platform overview.

## Storage Layer
We use DuckDB for vector search and structured analytics.

### DuckDB Configuration
The database file is stored in `./data/duckdb/store.duckdb`.

## Retrieval Pipeline
Hybrid retrieval fuses dense vectors and BM25 keywords.
"""
        docs = self.md_chunker.chunk_markdown(
            sample_md,
            title="System Architecture",
            source_url="https://wiki.corp.com/pages/123",
            doc_type="confluence",
        )
        self.assertGreaterEqual(len(docs), 3)

        # Verify hierarchical breadcrumbs
        breadcrumbs = [d.metadata.get("section") for d in docs]
        self.assertTrue(any("Storage Layer" in b for b in breadcrumbs))
        self.assertTrue(any("DuckDB Configuration" in b for b in breadcrumbs))

        # Check citation tag format
        for doc in docs:
            tag = doc.get_citation_tag()
            self.assertTrue(tag.startswith("[Source: Confluence - System Architecture"))


if __name__ == "__main__":
    unittest.main()
