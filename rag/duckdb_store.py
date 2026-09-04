"""
Embedded DuckDB storage engine holding vector embeddings, full-text indexes, and metadata.
Features:
- Schema: documents(id, content, embedding FLOAT[768], source_url, file_path, doc_type,
                    sprint_id, work_item_id, author, commit_hash, content_hash, created_at)
- Dense retrieval: array_cosine_similarity over FLOAT[768] arrays
- Full-Text Search (FTS): native BM25 scoring via DuckDB FTS extension
- SHA256 content_hash deduplication (skips unchanged documents)
- Strict memory capping via PRAGMA max_memory='2GB'
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb

from config import DUCKDB_PATH, EMBED_DIM, MAX_DUCKDB_MEMORY
from ingestion.base import Document


class DuckDBStore:
    """
    Persistent DuckDB database manager for RAG vectors, full-text search, and metadata.
    """

    def __init__(self, db_path: Optional[Path] = None, memory_limit: str = MAX_DUCKDB_MEMORY):
        self.db_path = Path(db_path or DUCKDB_PATH)
        self.memory_limit = memory_limit
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_indexed = False

        # Initialize schema
        self._init_db()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Opens a DuckDB connection with configured memory limit and extensions."""
        conn = duckdb.connect(str(self.db_path))
        conn.execute(f"PRAGMA max_memory='{self.memory_limit}';")
        conn.execute("PRAGMA threads=4;")
        try:
            conn.execute("LOAD fts;")
        except Exception:
            try:
                conn.execute("INSTALL fts; LOAD fts;")
            except Exception as e:
                print(f"Warning: Could not load DuckDB FTS extension: {e}")
        return conn

    def _init_db(self):
        """Creates the documents table if it doesn't already exist."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR PRIMARY KEY,
                    content TEXT,
                    embedding FLOAT[{EMBED_DIM}],
                    source_url VARCHAR,
                    file_path VARCHAR,
                    doc_type VARCHAR,
                    sprint_id VARCHAR,
                    work_item_id VARCHAR,
                    author VARCHAR,
                    commit_hash VARCHAR,
                    content_hash VARCHAR,
                    created_at TIMESTAMP,
                    metadata_json TEXT
                );
                """
            )
            # Create secondary index on content_hash for high-speed deduplication checks
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON documents (content_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_type ON documents (doc_type);")

    def filter_existing_hashes(self, content_hashes: List[str]) -> Set[str]:
        """
        Returns the subset of content_hashes that already exist in DuckDB.
        Used to prevent redundant Ollama embedding calculations.
        """
        if not content_hashes:
            return set()

        unique_hashes = list(set(content_hashes))
        with self._get_connection() as conn:
            # DuckDB handles array unnest or in-list queries quickly
            placeholders = ",".join(["?"] * len(unique_hashes))
            query = f"SELECT content_hash FROM documents WHERE content_hash IN ({placeholders});"
            rows = conn.execute(query, unique_hashes).fetchall()
            return {row[0] for row in rows}

    def insert_documents(self, docs: List[Document]) -> int:
        """
        Inserts documents with embeddings into DuckDB, skipping any whose content_hash exists.
        Returns the number of newly inserted documents.
        """
        if not docs:
            return 0

        # Filter out duplicates
        all_hashes = [d.content_hash for d in docs]
        existing_hashes = self.filter_existing_hashes(all_hashes)
        new_docs = [d for d in docs if d.content_hash not in existing_hashes]

        if not new_docs:
            return 0

        with self._get_connection() as conn:
            insert_sql = f"""
            INSERT INTO documents (
                id, content, embedding, source_url, file_path, doc_type,
                sprint_id, work_item_id, author, commit_hash, content_hash,
                created_at, metadata_json
            ) VALUES (?, ?, ?::FLOAT[{EMBED_DIM}], ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            data_tuples = []
            for d in new_docs:
                if d.embedding is None:
                    raise ValueError(f"Document {d.id} has no embedding.")
                data_tuples.append(
                    (
                        d.id,
                        d.content,
                        d.embedding,
                        d.source_url,
                        d.file_path,
                        d.doc_type,
                        d.sprint_id,
                        d.work_item_id,
                        d.author,
                        d.commit_hash,
                        d.content_hash,
                        d.created_at,
                        json.dumps(d.metadata),
                    )
                )

            conn.executemany(insert_sql, data_tuples)

        # Invalidate or rebuild FTS index
        self.rebuild_fts_index()
        return len(new_docs)

    def delete_documents_by_file(self, file_path: str, source_prefix: Optional[str] = None) -> int:
        """
        Deletes existing chunks for a specific file path.
        Used during incremental re-indexing when a file has been modified or deleted.
        """
        with self._get_connection() as conn:
            if source_prefix:
                sql = "DELETE FROM documents WHERE file_path = ? AND source_url LIKE ?;"
                cursor = conn.execute(sql, [file_path, f"%{source_prefix}%"])
            else:
                sql = "DELETE FROM documents WHERE file_path = ?;"
                cursor = conn.execute(sql, [file_path])
            deleted_count = cursor.fetchall()  # In duckdb, DELETE returns count or fetchall
            # Let's rebuild FTS if rows were deleted
            self.rebuild_fts_index()
            return 1

    def delete_documents_by_work_item(self, work_item_id: str) -> int:
        """
        Deletes existing chunks for an ADO work item or Confluence page ID.
        Used when an issue/page has been updated and re-indexed.
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE work_item_id = ?;", [str(work_item_id)])
            self.rebuild_fts_index()
            return 1

    def delete_documents_by_source(self, source_url: str) -> int:
        """Deletes all chunks originating from a specific source URL."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE source_url = ?;", [source_url])
            self.rebuild_fts_index()
            return 1

    def rebuild_fts_index(self):
        """Rebuilds the BM25 full-text index across all indexed document contents."""
        with self._get_connection() as conn:
            try:
                conn.execute("PRAGMA create_fts_index('documents', 'id', 'content', overwrite=1);")
                self._fts_indexed = True
            except Exception as e:
                # FTS index might fail if table is empty
                self._fts_indexed = False

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 25,
        doc_type: Optional[str] = None,
        sprint_id: Optional[str] = None,
        source_url_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Dense vector retrieval via brute-force cosine similarity over FLOAT[768].
        Applies SQL WHERE clauses for metadata/intent filtering.
        """
        where_clauses = ["embedding IS NOT NULL"]
        params: List[Any] = [query_vector]

        if doc_type and doc_type.lower() != "all":
            where_clauses.append("doc_type = ?")
            params.append(doc_type)
        if sprint_id:
            where_clauses.append("sprint_id = ?")
            params.append(sprint_id)
        if source_url_filter:
            where_clauses.append("source_url LIKE ?")
            params.append(f"%{source_url_filter}%")

        where_sql = " AND ".join(where_clauses)
        sql = f"""
        SELECT 
            id, content, source_url, file_path, doc_type,
            sprint_id, work_item_id, author, commit_hash, content_hash,
            created_at, metadata_json,
            array_cosine_similarity(embedding, ?::FLOAT[{EMBED_DIM}]) AS score
        FROM documents
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT {top_k};
        """

        results: List[Tuple[Document, float]] = []
        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            for r in rows:
                meta = json.loads(r[11]) if r[11] else {}
                doc = Document(
                    id=r[0],
                    content=r[1],
                    source_url=r[2],
                    file_path=r[3],
                    doc_type=r[4],
                    sprint_id=r[5],
                    work_item_id=r[6],
                    author=r[7],
                    commit_hash=r[8],
                    content_hash=r[9],
                    created_at=str(r[10]) if r[10] else None,
                    metadata=meta,
                )
                score = float(r[12]) if r[12] is not None else 0.0
                results.append((doc, score))

        return results

    def search_sparse(
        self,
        query_text: str,
        top_k: int = 25,
        doc_type: Optional[str] = None,
        sprint_id: Optional[str] = None,
        source_url_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Sparse keyword retrieval via DuckDB FTS (BM25).
        Falls back to token frequency / ILIKE if FTS index is unavailable.
        """
        clean_terms = [t for t in query_text.replace("'", " ").replace('"', " ").split() if len(t) > 1]
        if not clean_terms:
            return []

        where_clauses: List[str] = []
        params: List[Any] = []

        if doc_type and doc_type.lower() != "all":
            where_clauses.append("doc_type = ?")
            params.append(doc_type)
        if sprint_id:
            where_clauses.append("sprint_id = ?")
            params.append(sprint_id)
        if source_url_filter:
            where_clauses.append("source_url LIKE ?")
            params.append(f"%{source_url_filter}%")

        where_suffix = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""

        results: List[Tuple[Document, float]] = []
        with self._get_connection() as conn:
            # Try DuckDB FTS BM25 query
            fts_query_str = " ".join(clean_terms[:8])
            fts_sql = f"""
            SELECT 
                id, content, source_url, file_path, doc_type,
                sprint_id, work_item_id, author, commit_hash, content_hash,
                created_at, metadata_json,
                fts_main_documents.match_bm25(id, ?) AS score
            FROM documents
            WHERE fts_main_documents.match_bm25(id, ?) IS NOT NULL {where_suffix}
            ORDER BY score DESC
            LIMIT {top_k};
            """
            try:
                full_params = [fts_query_str, fts_query_str] + params
                rows = conn.execute(fts_sql, full_params).fetchall()
                for r in rows:
                    meta = json.loads(r[11]) if r[11] else {}
                    doc = Document(
                        id=r[0],
                        content=r[1],
                        source_url=r[2],
                        file_path=r[3],
                        doc_type=r[4],
                        sprint_id=r[5],
                        work_item_id=r[6],
                        author=r[7],
                        commit_hash=r[8],
                        content_hash=r[9],
                        created_at=str(r[10]) if r[10] else None,
                        metadata=meta,
                    )
                    score = float(r[12]) if r[12] is not None else 0.0
                    results.append((doc, score))
                if results:
                    return results
            except Exception:
                # If FTS index is missing or query syntax error, fall through to ILIKE fallback
                pass

            # Fallback ILIKE scoring if FTS extension is not ready
            term_scores = []
            for term in clean_terms[:5]:
                term_scores.append(f"(CASE WHEN content ILIKE '%{term}%' THEN 1 ELSE 0 END)")
            score_expr = " + ".join(term_scores) if term_scores else "1"

            fallback_where = f"({score_expr}) > 0 {where_suffix}"
            fallback_sql = f"""
            SELECT 
                id, content, source_url, file_path, doc_type,
                sprint_id, work_item_id, author, commit_hash, content_hash,
                created_at, metadata_json,
                ({score_expr}) AS score
            FROM documents
            WHERE {fallback_where}
            ORDER BY score DESC
            LIMIT {top_k};
            """
            try:
                rows = conn.execute(fallback_sql, params).fetchall()
                for r in rows:
                    meta = json.loads(r[11]) if r[11] else {}
                    doc = Document(
                        id=r[0],
                        content=r[1],
                        source_url=r[2],
                        file_path=r[3],
                        doc_type=r[4],
                        sprint_id=r[5],
                        work_item_id=r[6],
                        author=r[7],
                        commit_hash=r[8],
                        content_hash=r[9],
                        created_at=str(r[10]) if r[10] else None,
                        metadata=meta,
                    )
                    score = float(r[12]) if r[12] is not None else 0.0
                    results.append((doc, score))
            except Exception as e:
                print(f"Fallback search error: {e}")

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Returns statistics on indexed documents, breakdown by doc_type, and disk size."""
        stats: Dict[str, Any] = {
            "total_documents": 0,
            "by_type": {},
            "disk_size_mb": 0.0,
            "unique_sources": 0,
        }

        if self.db_path.exists():
            stats["disk_size_mb"] = round(self.db_path.stat().st_size / (1024 * 1024), 2)

        with self._get_connection() as conn:
            # Total docs
            row = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()
            stats["total_documents"] = row[0] if row else 0

            # Group by doc_type
            type_rows = conn.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type;").fetchall()
            stats["by_type"] = {r[0]: r[1] for r in type_rows}

            # Unique sources
            src_row = conn.execute("SELECT COUNT(DISTINCT source_url) FROM documents;").fetchone()
            stats["unique_sources"] = src_row[0] if src_row else 0

        return stats

    def clear_all(self):
        """Clears all indexed documents from the database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents;")
        self.rebuild_fts_index()
