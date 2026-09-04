"""
Data ingestion module for GitHub, Azure DevOps, and Confluence.
"""

from .base import Document, BaseLoader, compute_content_hash

__all__ = ["Document", "BaseLoader", "compute_content_hash"]
