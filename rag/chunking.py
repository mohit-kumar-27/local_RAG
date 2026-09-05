"""
Language-aware and hierarchy-preserving chunking module.
Supports AST-based chunking for Python, structural chunking for multi-language codebases,
and header-hierarchy-preserving Markdown chunking for Confluence and documentation.
Uses tiktoken with headroom for approximate token counting.
"""

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from config import CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS, MAX_CHUNK_TOKENS
from ingestion.base import Document

# Initialize tokenizer (cl100k_base used as robust BPE proxy with headroom)
_ENCODER = None

def get_tokenizer():
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return _ENCODER


def count_tokens(text: str) -> int:
    """
    Returns the approximate token count for a text snippet using tiktoken.
    Budgeted with headroom for Llama/Nomic tokenizers.
    """
    if not text:
        return 0
    try:
        enc = get_tokenizer()
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        # Fallback heuristic: 1 token ~= 4 characters in English/code
        return max(1, len(text) // 4)


def is_binary_file(file_path: Path, sample_size: int = 8192) -> bool:
    """Detects if a file is binary by inspecting null bytes and common binary extensions."""
    binary_extensions = {
        ".exe", ".dll", ".so", ".dylib", ".bin", ".iso",
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",
        ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
        ".pyc", ".pyd", ".pyo", ".class", ".jar", ".war",
        ".wasm", ".parquet", ".duckdb", ".sqlite", ".db",
        ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3"
    }
    if file_path.suffix.lower() in binary_extensions:
        return True
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
            if b"\x00" in chunk:
                return True
    except Exception:
        return True
    return False


def is_ignored_path(rel_path: str) -> bool:
    """Checks if a file or directory path should be skipped."""
    ignored_patterns = [
        r"(^|[/\\])\.git([/\\]|$)",
        r"(^|[/\\])\.svn([/\\]|$)",
        r"(^|[/\\])\.hg([/\\]|$)",
        r"(^|[/\\])node_modules([/\\]|$)",
        r"(^|[/\\])__pycache__([/\\]|$)",
        r"(^|[/\\])\.venv([/\\]|$)",
        r"(^|[/\\])venv([/\\]|$)",
        r"(^|[/\\])\.idea([/\\]|$)",
        r"(^|[/\\])\.vscode([/\\]|$)",
        r"(^|[/\\])dist([/\\]|$)",
        r"(^|[/\\])build([/\\]|$)",
        r"(^|[/\\])target([/\\]|$)",
        r"package-lock\.json$",
        r"yarn\.lock$",
        r"pnpm-lock\.yaml$",
        r"Cargo\.lock$",
        r"poetry\.lock$",
        r"\.min\.(js|css)$",
        r"\.map$",
        r"\.env($|\.)",
        r"\.pem$",
        r"\.key$",
        r"id_rsa",
    ]
    norm_path = rel_path.replace("\\", "/")
    for pattern in ignored_patterns:
        if re.search(pattern, norm_path, re.IGNORECASE):
            return True
    return False


class CodeChunker:
    """
    Code chunker supporting:
    1. AST-based chunking for Python (classes, functions, methods with signatures and docstrings).
    2. Structural chunking for other programming languages with line number tracking.
    """

    def __init__(
        self,
        target_tokens: int = CHUNK_TARGET_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        max_tokens: int = MAX_CHUNK_TOKENS,
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens

    def chunk_python_file(
        self, code: str, file_path: str, source_url: str
    ) -> List[Document]:
        """Splits Python source code into AST-derived logical chunks."""
        documents: List[Document] = []
        lines = code.splitlines()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback to generic structural chunking if AST parsing fails
            return self.chunk_generic_code(code, file_path, source_url)

        # Collect top-level imports & module docstrings for context
        module_doc = ast.get_docstring(tree) or ""
        header_context = f"# File: {file_path}\n"
        if module_doc:
            header_context += f'"""\n{module_doc.strip()}\n"""\n\n'

        ast_blocks: List[Tuple[int, int, str, str]] = []  # (start_line, end_line, name, kind)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ast_blocks.append((node.lineno, node.end_lineno or node.lineno, node.name, "function"))
            elif isinstance(node, ast.ClassDef):
                # For classes, check size. If small, keep whole class; if large, extract methods
                class_start = node.lineno
                class_end = node.end_lineno or node.lineno
                class_lines = "\n".join(lines[class_start - 1 : class_end])
                if count_tokens(class_lines) <= self.max_tokens:
                    ast_blocks.append((class_start, class_end, node.name, "class"))
                else:
                    # Split into class header + individual methods
                    methods_found = False
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods_found = True
                            ast_blocks.append(
                                (item.lineno, item.end_lineno or item.lineno, f"{node.name}.{item.name}", "method")
                            )
                    if not methods_found:
                        ast_blocks.append((class_start, class_end, node.name, "class"))

        # If no AST functions/classes were identified (e.g. flat script), fall back to generic
        if not ast_blocks:
            return self.chunk_generic_code(code, file_path, source_url)

        # Process AST blocks into Documents
        for start_line, end_line, name, kind in ast_blocks:
            block_content = "\n".join(lines[start_line - 1 : end_line])
            chunk_tokens = count_tokens(block_content)

            # If the block fits within limits, create document
            if chunk_tokens <= self.max_tokens:
                content = f"# File: {file_path}:{start_line}-{end_line} ({kind} {name})\n{block_content}"
                documents.append(
                    Document(
                        content=content,
                        source_url=source_url,
                        doc_type="code",
                        file_path=file_path,
                        metadata={
                            "start_line": start_line,
                            "end_line": end_line,
                            "symbol_name": name,
                            "symbol_type": kind,
                            "token_count": count_tokens(content),
                        },
                    )
                )
            else:
                # Sub-chunk large functions/classes using structural chunker
                sub_chunks = self.chunk_generic_code(
                    block_content,
                    file_path=file_path,
                    source_url=source_url,
                    line_offset=start_line - 1,
                    context_prefix=f"# Context: {kind} {name}\n",
                )
                documents.extend(sub_chunks)

        return documents

    def chunk_generic_code(
        self,
        code: str,
        file_path: str,
        source_url: str,
        line_offset: int = 0,
        context_prefix: str = "",
    ) -> List[Document]:
        """
        Splits arbitrary source code (JS, TS, Go, C#, Java, etc.) into overlapping token-budgeted chunks
        while maintaining line numbers and file headers.
        """
        documents: List[Document] = []
        raw_lines = code.splitlines()
        if not raw_lines:
            return documents

        # Sanitize lines: break any gigantic lines (e.g. minified JS, base64 strings) so no chunk can blow up
        max_line_chars = max(500, self.max_tokens * 3)
        lines: List[str] = []
        for l in raw_lines:
            if len(l) > max_line_chars:
                for i in range(0, len(l), max_line_chars):
                    lines.append(l[i : i + max_line_chars])
            else:
                lines.append(l)

        current_lines: List[str] = []
        start_idx = 0

        for idx, line in enumerate(lines):
            current_lines.append(line)
            current_text = "\n".join(current_lines)
            tokens = count_tokens(current_text)

            if tokens >= self.target_tokens or idx == len(lines) - 1:
                start_line = line_offset + start_idx + 1
                end_line = line_offset + idx + 1
                header = f"# File: {file_path}:{start_line}-{end_line}\n"
                if context_prefix:
                    header += context_prefix

                full_chunk_text = header + current_text
                documents.append(
                    Document(
                        content=full_chunk_text,
                        source_url=source_url,
                        doc_type="code",
                        file_path=file_path,
                        metadata={
                            "start_line": start_line,
                            "end_line": end_line,
                            "token_count": count_tokens(full_chunk_text),
                        },
                    )
                )

                # Compute line overlap for smooth context continuity
                overlap_lines: List[str] = []
                overlap_token_count = 0
                for r_line in reversed(current_lines):
                    overlap_lines.insert(0, r_line)
                    overlap_token_count = count_tokens("\n".join(overlap_lines))
                    if overlap_token_count >= self.overlap_tokens:
                        break

                current_lines = overlap_lines
                start_idx = idx - len(overlap_lines) + 1

        return documents

    def chunk_notebook(self, content: str, file_path: str, source_url: str) -> List[Document]:
        """
        Parses Jupyter Notebook (.ipynb) files:
        - Extracts markdown documentation cells and Python code cells.
        - Completely strips execution outputs, base64 images, and raw stdout/stderr blobs.
        - Chunks code cells and markdown cells cleanly with line/cell context.
        """
        import json
        documents: List[Document] = []
        try:
            nb = json.loads(content)
        except Exception:
            # Fallback if notebook is malformed
            return self.chunk_generic_code(content, file_path, source_url)

        cells = nb.get("cells", [])
        for idx, cell in enumerate(cells, start=1):
            cell_type = cell.get("cell_type", "")
            if cell_type not in ("code", "markdown"):
                continue

            raw_source = cell.get("source", "")
            if isinstance(raw_source, list):
                source_text = "".join(raw_source)
            else:
                source_text = str(raw_source)

            source_text = source_text.strip()
            if not source_text:
                continue

            cell_tokens = count_tokens(source_text)
            if cell_tokens <= self.max_tokens:
                header = f"# Notebook: {file_path} (Cell {idx} - {cell_type})\n"
                doc_content = header + source_text
                documents.append(
                    Document(
                        content=doc_content,
                        source_url=source_url,
                        doc_type="code" if cell_type == "code" else "confluence",
                        file_path=file_path,
                        metadata={
                            "cell_index": idx,
                            "cell_type": cell_type,
                            "token_count": count_tokens(doc_content),
                        },
                    )
                )
            else:
                # Sub-chunk oversized cell
                sub_chunks = self.chunk_generic_code(
                    source_text,
                    file_path=file_path,
                    source_url=source_url,
                    context_prefix=f"# Context: Notebook {file_path} (Cell {idx} - {cell_type})\n",
                )
                documents.extend(sub_chunks)

        return documents

    def chunk_file(self, content: str, file_path: str, source_url: str) -> List[Document]:
        """Routes file content to notebook parser, AST chunker (if Python), or generic structural chunker."""
        lower = file_path.lower()
        if lower.endswith(".ipynb"):
            return self.chunk_notebook(content, file_path, source_url)
        elif lower.endswith(".py"):
            return self.chunk_python_file(content, file_path, source_url)
        else:
            return self.chunk_generic_code(content, file_path, source_url)


class MarkdownSectionChunker:
    """
    Hierarchy-preserving Markdown chunker designed for Confluence and technical documentation.
    Maintains breadcrumb headers (# Title > ## Section > ### Subsection) so each chunk
    retains full contextual meaning when retrieved independently.
    """

    def __init__(
        self,
        target_tokens: int = CHUNK_TARGET_TOKENS,
        max_tokens: int = MAX_CHUNK_TOKENS,
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        # Regex matching markdown headers: # Header 1, ## Header 2, etc.
        self.header_regex = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def chunk_markdown(
        self,
        markdown_text: str,
        title: str,
        source_url: str,
        doc_type: str = "confluence",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Parses markdown text into hierarchical section chunks.
        """
        documents: List[Document] = []
        if not markdown_text.strip():
            return documents

        lines = markdown_text.splitlines()
        # Stack of (level, title)
        header_stack: List[Tuple[int, str]] = []

        current_section_lines: List[str] = []
        current_section_title = title

        def flush_section(lines_to_flush: List[str], current_title: str):
            if not lines_to_flush:
                return
            body = "\n".join(lines_to_flush).strip()
            if not body:
                return

            breadcrumb = " > ".join(t for _, t in header_stack) if header_stack else title
            header_prefix = f"# Document: {title}\n# Section: {breadcrumb}\n\n"

            # Check if section exceeds max tokens
            total_tokens = count_tokens(header_prefix + body)
            if total_tokens <= self.max_tokens:
                doc_content = header_prefix + body
                meta = {
                    "title": title,
                    "section": breadcrumb,
                    "token_count": count_tokens(doc_content),
                }
                if extra_metadata:
                    meta.update(extra_metadata)

                documents.append(
                    Document(
                        content=doc_content,
                        source_url=source_url,
                        doc_type=doc_type,
                        metadata=meta,
                    )
                )
            else:
                # Sub-split large section by paragraphs
                paragraphs = body.split("\n\n")
                sub_lines: List[str] = []
                for p in paragraphs:
                    sub_lines.append(p)
                    sub_text = "\n\n".join(sub_lines)
                    if count_tokens(header_prefix + sub_text) >= self.target_tokens:
                        doc_content = header_prefix + sub_text
                        meta = {
                            "title": title,
                            "section": breadcrumb,
                            "token_count": count_tokens(doc_content),
                        }
                        if extra_metadata:
                            meta.update(extra_metadata)

                        documents.append(
                            Document(
                                content=doc_content,
                                source_url=source_url,
                                doc_type=doc_type,
                                metadata=meta,
                            )
                        )
                        sub_lines = []

                if sub_lines:
                    sub_text = "\n\n".join(sub_lines)
                    doc_content = header_prefix + sub_text
                    meta = {
                        "title": title,
                        "section": breadcrumb,
                        "token_count": count_tokens(doc_content),
                    }
                    if extra_metadata:
                        meta.update(extra_metadata)

                    documents.append(
                        Document(
                            content=doc_content,
                            source_url=source_url,
                            doc_type=doc_type,
                            metadata=meta,
                        )
                    )

        for line in lines:
            match = self.header_regex.match(line)
            if match:
                # Flush previous section
                flush_section(current_section_lines, current_section_title)
                current_section_lines = []

                level = len(match.group(1))
                h_title = match.group(2).strip()

                # Update breadcrumb stack
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, h_title))
                current_section_title = h_title
                current_section_lines.append(line)
            else:
                current_section_lines.append(line)

        # Flush final section
        flush_section(current_section_lines, current_section_title)

        return documents
