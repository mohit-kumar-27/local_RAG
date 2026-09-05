"""
Azure DevOps (ADO) loader for Boards (Work Items, Epics, Stories, Bugs, Tasks, Sprints, Comments)
and Git Repositories using httpx.AsyncClient and shallow git clones.
"""

import base64
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from config import ADO_ORGANIZATION, ADO_PAT, ADO_PROJECT, REPOS_CACHE_DIR
from ingestion.base import BaseLoader, Document
from rag.chunking import CodeChunker, is_binary_file, is_ignored_path


def html_to_plain_text(html_content: str) -> str:
    """Strips HTML tags and standardizes whitespace for ADO rich-text fields."""
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()
    except Exception:
        # Fallback regex strip
        clean = re.sub(r"<[^>]+>", " ", html_content)
        return " ".join(clean.split()).strip()


def parse_ado_url(url: str) -> Dict[str, str]:
    """
    Parses ADO URL to extract organization, project, and component.
    Supports formats:
      - https://dev.azure.com/{org}/{project}/_workitems/edit/{id}
      - https://dev.azure.com/{org}/{project}/_boards
      - https://dev.azure.com/{org}/{project}/_git/{repo}
      - https://{org}.visualstudio.com/{project}
    """
    parsed = urlparse(url.strip())
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]

    result: Dict[str, str] = {
        "organization": ADO_ORGANIZATION,
        "project": ADO_PROJECT,
        "type": "board",
        "repo": "",
        "work_item_id": "",
    }

    if "dev.azure.com" in parsed.netloc and len(path_parts) >= 2:
        result["organization"] = path_parts[0]
        result["project"] = path_parts[1]
        if len(path_parts) >= 4 and path_parts[2] == "_git":
            result["type"] = "repo"
            result["repo"] = path_parts[3]
        elif len(path_parts) >= 4 and path_parts[2] == "_workitems" and path_parts[3] == "edit":
            result["type"] = "work_item"
            result["work_item_id"] = path_parts[4] if len(path_parts) > 4 else ""
    elif "visualstudio.com" in parsed.netloc:
        result["organization"] = parsed.netloc.split(".")[0]
        if path_parts:
            result["project"] = path_parts[0]

    return result


class AdoBoardLoader(BaseLoader):
    """
    Extracts Work Items (Epics, Stories, Bugs, Tasks), Sprints, Acceptance Criteria,
    and Discussion comments from Azure DevOps Boards using httpx.AsyncClient.
    """

    def __init__(
        self,
        board_url_or_query: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        pat_token: Optional[str] = None,
        max_items: int = 100,
    ):
        self.raw_url = board_url_or_query or ""
        parsed_info = parse_ado_url(self.raw_url) if self.raw_url else {}

        self.org = organization or parsed_info.get("organization") or ADO_ORGANIZATION
        self.project = project or parsed_info.get("project") or ADO_PROJECT
        self.pat = pat_token or ADO_PAT
        self.max_items = max_items

        if not self.org or not self.project:
            raise ValueError("Azure DevOps organization and project must be configured.")

        # HTTP Basic Auth: ADO expects base64 encoded ":{PAT}"
        auth_bytes = f":{self.pat}".encode("ascii")
        self.auth_header = f"Basic {base64.b64encode(auth_bytes).decode('ascii')}"
        self.base_api_url = f"https://dev.azure.com/{self.org}/{self.project}/_apis"

    async def _query_work_item_ids(self, client: httpx.AsyncClient) -> List[int]:
        """Runs a WIQL query to retrieve recent work item IDs."""
        wiql_url = f"{self.base_api_url}/wit/wiql?api-version=7.1-preview.2"
        wiql_query = {
            "query": (
                "SELECT [System.Id] "
                "FROM WorkItems "
                f"WHERE [System.TeamProject] = '{self.project}' "
                "ORDER BY [System.ChangedDate] DESC"
            )
        }
        res = await client.post(wiql_url, json=wiql_query, headers={"Authorization": self.auth_header})
        if res.status_code != 200:
            raise RuntimeError(f"ADO WIQL query failed ({res.status_code}): {res.text}")

        data = res.json()
        work_items = data.get("workItems", [])
        return [item["id"] for item in work_items[: self.max_items]]

    async def _fetch_comments(self, client: httpx.AsyncClient, work_item_id: int) -> List[str]:
        """Fetches discussion comments for a work item."""
        comments_url = f"{self.base_api_url}/wit/workItems/{work_item_id}/comments?api-version=7.1-preview.3&$top=5"
        try:
            res = await client.get(comments_url, headers={"Authorization": self.auth_header})
            if res.status_code == 200:
                comments_data = res.json().get("comments", [])
                extracted = []
                for c in comments_data:
                    author = c.get("createdBy", {}).get("displayName", "User")
                    text = html_to_plain_text(c.get("text", ""))
                    if text:
                        extracted.append(f"{author}: {text}")
                return extracted
        except Exception:
            pass
        return []

    async def load(self) -> List[Document]:
        """Fetches work items, cleans description & acceptance criteria, and constructs Documents."""
        documents: List[Document] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            ids = await self._query_work_item_ids(client)
            if not ids:
                return documents

            # Batch retrieve work items in chunks of 50
            chunk_size = 50
            for i in range(0, len(ids), chunk_size):
                batch_ids = ids[i : i + chunk_size]
                ids_param = ",".join(str(wid) for wid in batch_ids)
                items_url = f"{self.base_api_url}/wit/workitems?ids={ids_param}&$expand=all&api-version=7.1-preview.3"

                res = await client.get(items_url, headers={"Authorization": self.auth_header})
                if res.status_code != 200:
                    continue

                items = res.json().get("value", [])
                for item in items:
                    wid = str(item.get("id"))
                    fields = item.get("fields", {})

                    wi_type = fields.get("System.WorkItemType", "Work Item")
                    title = fields.get("System.Title", "Untitled")
                    state = fields.get("System.State", "Unknown")
                    assigned_to = fields.get("System.AssignedTo", {}).get("displayName", "Unassigned")
                    iteration = fields.get("System.IterationPath", "")
                    description = html_to_plain_text(fields.get("System.Description", ""))
                    criteria = html_to_plain_text(fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", ""))
                    author = fields.get("System.CreatedBy", {}).get("displayName", "Unknown")

                    # Fetch discussion comments
                    comments = await self._fetch_comments(client, int(wid))
                    comments_str = "\n".join(f"- {c}" for c in comments) if comments else "None"

                    source_url = f"https://dev.azure.com/{self.org}/{self.project}/_workitems/edit/{wid}"

                    content = (
                        f"# Azure DevOps {wi_type} #{wid}: {title}\n"
                        f"State: {state} | Assigned To: {assigned_to} | Sprint/Iteration: {iteration}\n"
                        f"URL: {source_url}\n\n"
                        f"## Description\n{description or 'No description provided.'}\n\n"
                    )
                    if criteria:
                        content += f"## Acceptance Criteria\n{criteria}\n\n"
                    if comments:
                        content += f"## Discussion / Comments\n{comments_str}\n"

                    documents.append(
                        Document(
                            content=content,
                            source_url=source_url,
                            doc_type="ticket",
                            work_item_id=wid,
                            sprint_id=iteration,
                            author=author,
                            metadata={
                                "type": wi_type,
                                "title": title,
                                "state": state,
                                "assigned_to": assigned_to,
                                "iteration": iteration,
                                "source": "azure_devops",
                            },
                        )
                    )

        return documents


class AdoRepoLoader(BaseLoader):
    """
    Azure DevOps Git repository loader.
    Clones repos using PAT authentication and applies AST/code chunking.
    """

    def __init__(
        self,
        repo_url: str,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        repo_name: Optional[str] = None,
        pat_token: Optional[str] = None,
        branch: str = "main",
    ):
        self.repo_url = repo_url
        parsed = parse_ado_url(repo_url)
        self.org = organization or parsed.get("organization") or ADO_ORGANIZATION
        self.project = project or parsed.get("project") or ADO_PROJECT
        self.repo_name = repo_name or parsed.get("repo") or "repo"
        self.pat = pat_token or ADO_PAT
        self.branch = branch

    def _get_clone_url(self) -> str:
        if self.pat:
            return f"https://{self.pat}@dev.azure.com/{self.org}/{self.project}/_git/{self.repo_name}"
        return f"https://dev.azure.com/{self.org}/{self.project}/_git/{self.repo_name}"

    async def load(self) -> List[Document]:
        """Shallow clones ADO repository and runs language-aware chunking."""
        target_dir = REPOS_CACHE_DIR / f"ado_{self.org}_{self.project}_{self.repo_name}"
        target_dir.mkdir(parents=True, exist_ok=True)
        clone_url = self._get_clone_url()

        if not (target_dir / ".git").exists():
            cmd = ["git", "clone", "--depth", "1", "--branch", self.branch, clone_url, str(target_dir)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                # Fallback without explicit branch
                fallback_cmd = ["git", "clone", "--depth", "1", clone_url, str(target_dir)]
                subprocess.run(fallback_cmd, capture_output=True, text=True, check=True, timeout=120)
        else:
            try:
                subprocess.run(["git", "pull", "--depth", "1"], cwd=str(target_dir), capture_output=True, text=True, timeout=60)
            except Exception:
                pass

        chunker = CodeChunker()
        documents: List[Document] = []

        for root, dirs, files in os.walk(target_dir):
            rel_root = os.path.relpath(root, target_dir)
            if is_ignored_path(rel_root):
                continue
            for fname in files:
                rel_file = os.path.normpath(os.path.join(rel_root, fname))
                if is_ignored_path(rel_file):
                    continue
                abs_file = Path(root) / fname
                if is_binary_file(abs_file):
                    continue
                try:
                    with open(abs_file, "r", encoding="utf-8", errors="replace") as f:
                        code = f.read()
                    source_url = f"https://dev.azure.com/{self.org}/{self.project}/_git/{self.repo_name}?path={rel_file.replace(chr(92), '/')}"
                    docs = chunker.chunk_file(code, file_path=rel_file, source_url=source_url)
                    documents.extend(docs)
                except Exception:
                    continue

        return documents

    def load_incremental(
        self,
        store: Optional[Any] = None,
    ) -> Tuple[List[Document], List[str], List[str]]:
        """
        Incrementally pulls ADO repository updates and processes ONLY modified, added, and deleted files.
        Deletes stale chunks for modified/deleted files and re-chunks only new content.
        Returns: (new_or_modified_documents, modified_files, deleted_files)
        """
        target_dir = REPOS_CACHE_DIR / f"ado_{self.org}_{self.project}_{self.repo_name}"
        target_dir.mkdir(parents=True, exist_ok=True)
        clone_url = self._get_clone_url()

        if not (target_dir / ".git").exists():
            # First time: full clone
            cmd = ["git", "clone", "--depth", "1", "--branch", self.branch, clone_url, str(target_dir)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                fallback_cmd = ["git", "clone", "--depth", "1", clone_url, str(target_dir)]
                subprocess.run(fallback_cmd, capture_output=True, text=True, check=True, timeout=120)
            return asyncio.run(self.load()) if not asyncio.get_event_loop().is_running() else [], [], []

        old_commit = "head"
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(target_dir), capture_output=True, text=True)
            if res.returncode == 0:
                old_commit = res.stdout.strip()
        except Exception:
            pass

        try:
            subprocess.run(["git", "pull", "--depth", "1"], cwd=str(target_dir), capture_output=True, text=True, timeout=60)
        except Exception:
            pass

        new_commit = old_commit
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(target_dir), capture_output=True, text=True)
            if res.returncode == 0:
                new_commit = res.stdout.strip()
        except Exception:
            pass

        if old_commit == new_commit and old_commit != "head":
            return [], [], []

        modified_files: List[str] = []
        deleted_files: List[str] = []
        added_files: List[str] = []

        try:
            diff_res = subprocess.run(
                ["git", "diff", "--name-status", old_commit, new_commit],
                cwd=str(target_dir),
                capture_output=True,
                text=True,
            )
            if diff_res.returncode == 0:
                for line in diff_res.stdout.splitlines():
                    parts = line.strip().split("\t")
                    if not parts:
                        continue
                    status = parts[0]
                    file_name = parts[1] if len(parts) > 1 else ""
                    if is_ignored_path(file_name):
                        continue
                    if status.startswith("M"):
                        modified_files.append(file_name)
                    elif status.startswith("D"):
                        deleted_files.append(file_name)
                    elif status.startswith("A"):
                        added_files.append(file_name)
        except Exception:
            pass

        if store:
            for d_file in deleted_files + modified_files:
                norm_d_file = os.path.normpath(d_file)
                store.delete_documents_by_file(self.repo_url, norm_d_file)

        chunker = CodeChunker()
        new_docs: List[Document] = []
        for file_to_chunk in modified_files + added_files:
            abs_path = target_dir / file_to_chunk
            if not abs_path.exists() or is_binary_file(abs_path):
                continue
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()
                norm_rel = os.path.normpath(file_to_chunk)
                source_url = f"https://dev.azure.com/{self.org}/{self.project}/_git/{self.repo_name}?path={norm_rel.replace(chr(92), '/')}"
                docs = chunker.chunk_file(code, file_path=norm_rel, source_url=source_url)
                new_docs.extend(docs)
            except Exception:
                continue

        return new_docs, modified_files, deleted_files
