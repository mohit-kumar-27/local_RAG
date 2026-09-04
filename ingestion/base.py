"""
Base document schemas, hashing utilities, and loader abstractions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional
import uuid


def compute_content_hash(text: str) -> str:
    """
    Computes a deterministic SHA256 content hash of normalized text for deduplication.
    Normalized by stripping leading/trailing whitespace and standardizing line breaks.
    """
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class Document:
    """
    Unified representation of an indexed chunk across all sources:
    GitHub Code/Issues, Azure DevOps Work Items/Repos, and Confluence Wiki Pages.
    Directly aligns with the DuckDB 'documents' schema.
    """
    content: str
    source_url: str
    doc_type: str  # 'code', 'ticket', 'confluence'
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Optional[str] = None
    sprint_id: Optional[str] = None
    work_item_id: Optional[str] = None
    author: Optional[str] = None
    commit_hash: Optional[str] = None
    content_hash: str = ""
    created_at: Optional[str] = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = compute_content_hash(self.content)

    def get_citation_tag(self) -> str:
        """
        Produces the standardized inline citation tag for this document.
        e.g., [Source: auth_service.py:45]
              [Source: ADO Bug #1234 - Login Failure]
              [Source: Confluence - Architecture Guide]
        """
        if self.doc_type == "code":
            line_str = f":{self.metadata.get('start_line', 1)}" if "start_line" in self.metadata else ""
            fname = self.file_path or self.source_url
            return f"[Source: {fname}{line_str}]"
        elif self.doc_type == "ticket":
            item_type = self.metadata.get("type", "Work Item")
            wid = self.work_item_id or self.id
            title = self.metadata.get("title", "")
            title_str = f" - {title}" if title else ""
            return f"[Source: ADO {item_type} #{wid}{title_str}]"
        elif self.doc_type == "confluence":
            title = self.metadata.get("title", "Page")
            section = self.metadata.get("section", "")
            sec_str = f" > {section}" if section else ""
            return f"[Source: Confluence - {title}{sec_str}]"
        else:
            return f"[Source: {self.source_url}]"


class BaseLoader(ABC):
    """Abstract interface for all data source loaders."""

    @abstractmethod
    async def load(self) -> List[Document]:
        """Asynchronously extracts and chunks documents from the source."""
        pass
