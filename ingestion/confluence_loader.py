"""
Atlassian Confluence loader using httpx.AsyncClient, toolslm HTML-to-Markdown conversion,
and hierarchy-preserving section chunking.
"""

import base64
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import httpx
from toolslm.download import html2md

from config import CONFLUENCE_API_TOKEN, CONFLUENCE_EMAIL, CONFLUENCE_URL
from ingestion.base import BaseLoader, Document
from rag.chunking import MarkdownSectionChunker


def parse_confluence_url(url: str) -> Dict[str, str]:
    """
    Extracts base_url, space_key, and page_id from standard Confluence URLs.
    Supports:
      - https://org.atlassian.net/wiki/spaces/ENG/pages/123456/Architecture
      - https://org.atlassian.net/wiki/spaces/ENG
      - https://confluence.company.com/pages/viewpage.action?pageId=123456
      - https://confluence.company.com/display/ENG/Architecture
    """
    url = url.strip()
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    if "/wiki" in parsed.path:
        base_url += "/wiki"

    result = {
        "base_url": base_url or CONFLUENCE_URL,
        "space_key": "",
        "page_id": "",
    }

    # Query param pageId (common in Server/Data Center)
    qs = parse_qs(parsed.query)
    if "pageId" in qs:
        result["page_id"] = qs["pageId"][0]

    path_parts = [p for p in parsed.path.split("/") if p]
    for i, part in enumerate(path_parts):
        if part.lower() == "spaces" and i + 1 < len(path_parts):
            result["space_key"] = path_parts[i + 1]
        elif part.lower() == "pages" and i + 1 < len(path_parts):
            if path_parts[i + 1].isdigit():
                result["page_id"] = path_parts[i + 1]
        elif part.lower() == "display" and i + 1 < len(path_parts):
            result["space_key"] = path_parts[i + 1]

    return result


class ConfluenceLoader(BaseLoader):
    """
    Fetches Confluence wiki pages via REST API, converts HTML to Markdown with toolslm,
    and applies header-hierarchy chunking.
    """

    def __init__(
        self,
        url_or_space: str,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        max_pages: int = 50,
    ):
        parsed = parse_confluence_url(url_or_space)
        self.base_url = (base_url or parsed.get("base_url") or CONFLUENCE_URL).rstrip("/")
        self.space_key = parsed.get("space_key") or ""
        self.page_id = parsed.get("page_id") or ""
        self.email = email or CONFLUENCE_EMAIL
        self.token = api_token or CONFLUENCE_API_TOKEN
        self.max_pages = max_pages
        self.chunker = MarkdownSectionChunker()

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.email and self.token:
            # Atlassian Cloud Basic Auth
            auth_str = f"{self.email}:{self.token}"
            encoded = base64.b64encode(auth_str.encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {encoded}"
        elif self.token:
            # Bearer Token / PAT for Data Center
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _fetch_single_page(self, client: httpx.AsyncClient, page_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single page by ID with storage format body and version info."""
        api_url = f"{self.base_url}/rest/api/content/{page_id}?expand=body.storage,version,ancestors,space"
        try:
            res = await client.get(api_url, headers=self._get_headers())
            if res.status_code == 200:
                return res.json()
            else:
                print(f"Failed to fetch Confluence page {page_id}: HTTP {res.status_code}")
        except Exception as e:
            print(f"Error fetching Confluence page {page_id}: {e}")
        return None

    async def _fetch_space_pages(self, client: httpx.AsyncClient, space_key: str) -> List[Dict[str, Any]]:
        """Fetches up to max_pages from a specific space."""
        pages: List[Dict[str, Any]] = []
        start = 0
        limit = min(25, self.max_pages)

        while len(pages) < self.max_pages:
            api_url = (
                f"{self.base_url}/rest/api/content"
                f"?spaceKey={space_key}&type=page&status=current"
                f"&start={start}&limit={limit}&expand=body.storage,version,ancestors,space"
            )
            try:
                res = await client.get(api_url, headers=self._get_headers())
                if res.status_code != 200:
                    break
                data = res.json()
                results = data.get("results", [])
                if not results:
                    break
                pages.extend(results)
                if len(results) < limit:
                    break
                start += len(results)
            except Exception as e:
                print(f"Error fetching space {space_key}: {e}")
                break

        return pages[: self.max_pages]

    async def load(self) -> List[Document]:
        """Loads and chunks Confluence pages."""
        documents: List[Document] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            pages_data: List[Dict[str, Any]] = []

            if self.page_id:
                single = await self._fetch_single_page(client, self.page_id)
                if single:
                    pages_data.append(single)
            elif self.space_key:
                pages_data = await self._fetch_space_pages(client, self.space_key)
            else:
                raise ValueError("Either a valid page ID or space key must be provided.")

            for page in pages_data:
                page_id = str(page.get("id"))
                title = page.get("title", "Untitled Page")
                version = str(page.get("version", {}).get("number", "1"))
                author = page.get("version", {}).get("by", {}).get("displayName", "Unknown")
                space_info = page.get("space", {})
                space_name = space_info.get("name") or space_info.get("key") or self.space_key

                # Web link
                links = page.get("_links", {})
                web_url = f"{self.base_url}{links.get('webui', f'/spaces/{self.space_key}/pages/{page_id}')}"

                # Body HTML in Atlassian storage format
                body_html = page.get("body", {}).get("storage", {}).get("value", "")
                if not body_html:
                    continue

                # Convert HTML to clean Markdown with toolslm
                try:
                    markdown = html2md(body_html)
                except Exception:
                    # Fallback plain text if markdown conversion fails
                    markdown = body_html

                # Hierarchical section chunking
                page_docs = self.chunker.chunk_markdown(
                    markdown_text=markdown,
                    title=title,
                    source_url=web_url,
                    doc_type="confluence",
                    extra_metadata={
                        "page_id": page_id,
                        "space": space_name,
                        "version": version,
                        "author": author,
                    },
                )
                for d in page_docs:
                    d.author = author
                    d.work_item_id = page_id

                documents.extend(page_docs)

        return documents
