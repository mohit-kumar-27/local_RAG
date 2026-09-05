"""
GitHub loader combining ghapi (metadata, issues, PRs, trees)
and shallow git clone for high-throughput, rate-limit-free bulk source ingestion.
"""

import inspect
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastcore.parallel import parallel
from ghapi.all import GhApi

from config import GITHUB_PAT, REPOS_CACHE_DIR
from ingestion.base import BaseLoader, Document
from rag.chunking import CodeChunker, is_binary_file, is_ignored_path


def parse_github_url(url: str) -> Tuple[str, str, Optional[str]]:
    """
    Parses a GitHub URL into (owner, repo, branch_or_path).
    Supports:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - https://github.com/owner/repo/tree/main
      - owner/repo
    """
    url = url.strip()
    if "/" in url and not url.startswith("http"):
        parts = url.split("/")
        return parts[0], parts[1].replace(".git", ""), None

    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {url}")

    owner = path_parts[0]
    repo = path_parts[1].replace(".git", "")
    branch = None
    if len(path_parts) >= 4 and path_parts[2] in ("tree", "blob"):
        branch = path_parts[3]

    return owner, repo, branch


def _read_and_chunk_file_worker(args: Tuple[str, str, str, str, str]) -> List[Document]:
    """Helper worker for fastcore.parallel file reading and chunking."""
    abs_path_str, rel_path_str, repo_url, branch, commit_hash = args
    abs_path = Path(abs_path_str)

    if is_binary_file(abs_path):
        return []

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return []

    chunker = CodeChunker()
    # Build web URL for direct linking
    source_url = f"{repo_url}/blob/{branch}/{rel_path_str.replace(chr(92), '/')}"
    docs = chunker.chunk_file(content, file_path=rel_path_str, source_url=source_url)
    for doc in docs:
        doc.commit_hash = commit_hash
        doc.author = "git"
    return docs


class GithubCodeLoader(BaseLoader):
    """
    High-throughput GitHub loader:
    - ghapi for authenticated metadata, README, and Issues.
    - Shallow git clone for bulk source code files (avoiding thousands of REST calls).
    """

    def __init__(
        self,
        repo_url_or_slug: str,
        pat_token: Optional[str] = None,
        branch: Optional[str] = None,
        include_issues: bool = True,
        max_files: Optional[int] = None,
    ):
        self.repo_url = repo_url_or_slug.strip()
        self.token = pat_token or GITHUB_PAT or None
        self.owner, self.repo, parsed_branch = parse_github_url(self.repo_url)
        self.branch = branch or parsed_branch or "main"
        self.include_issues = include_issues
        self.max_files = max_files

        # Initialize ghapi
        self.api = GhApi(owner=self.owner, repo=self.repo, token=self.token)

    def _get_clone_url(self) -> str:
        if self.token:
            return f"https://x-access-token:{self.token}@github.com/{self.owner}/{self.repo}.git"
        return f"https://github.com/{self.owner}/{self.repo}.git"

    def _shallow_clone_or_update(self, target_dir: Path) -> Tuple[bool, str]:
        """Performs a shallow git clone (--depth 1) or fast forward fetch."""
        target_dir.mkdir(parents=True, exist_ok=True)
        clone_url = self._get_clone_url()
        commit_hash = "head"

        if (target_dir / ".git").exists():
            # Already cloned, fetch latest shallow commit
            try:
                subprocess.run(
                    ["git", "pull", "--depth", "1"],
                    cwd=str(target_dir),
                    capture_output=True,
                    check=True,
                    timeout=60,
                )
            except Exception:
                pass
        else:
            cmd = ["git", "clone", "--depth", "1", "--branch", self.branch, clone_url, str(target_dir)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                # Fall back to cloning default branch without explicit branch flag
                fallback_cmd = ["git", "clone", "--depth", "1", clone_url, str(target_dir)]
                fallback_res = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=120)
                if fallback_res.returncode != 0:
                    raise RuntimeError(f"Git clone failed: {fallback_res.stderr or res.stderr}")

        # Extract current commit hash
        try:
            hash_res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if hash_res.returncode == 0:
                commit_hash = hash_res.stdout.strip()
        except Exception:
            pass

        return True, commit_hash

    async def _load_issues(self) -> List[Document]:
        """Fetches issues and discussions via ghapi."""
        documents: List[Document] = []
        try:
            res = self.api.issues.list_for_repo(state="all", per_page=50)
            if inspect.isawaitable(res):
                issues = await res
            else:
                issues = res
            for issue in issues:
                if getattr(issue, "pull_request", None):
                    # Skip PRs if we only want issues, or index them as PR
                    doc_kind = "Pull Request"
                else:
                    doc_kind = "Issue"

                issue_num = issue.number
                title = issue.title or ""
                body = issue.body or ""
                state = issue.state
                author = issue.user.login if issue.user else "unknown"
                html_url = issue.html_url

                content = (
                    f"# GitHub {doc_kind} #{issue_num}: {title}\n"
                    f"Status: {state} | Author: {author}\n"
                    f"URL: {html_url}\n\n"
                    f"## Description\n{body}\n"
                )

                documents.append(
                    Document(
                        content=content,
                        source_url=html_url,
                        doc_type="ticket",
                        work_item_id=str(issue_num),
                        author=author,
                        metadata={
                            "type": doc_kind,
                            "title": title,
                            "number": issue_num,
                            "state": state,
                            "source": "github",
                        },
                    )
                )
        except Exception as e:
            # Handle rate limits or missing scopes gracefully
            print(f"Warning: could not fetch GitHub issues via API: {e}")
        return documents

    def load_incremental(
        self,
        store: Optional[Any] = None,
    ) -> Tuple[List[Document], List[str], List[str]]:
        """
        Incrementally pulls repository updates and processes ONLY modified, added, and deleted files.
        - Deletes stale chunks for modified and deleted files from DuckDB.
        - Chunks and yields ONLY modified and added files for embedding.
        - If repository is up-to-date, completes in milliseconds with zero redundant compute.
        Returns: (new_or_modified_documents, modified_files, deleted_files)
        """
        target_dir = REPOS_CACHE_DIR / f"{self.owner}_{self.repo}"
        target_dir.mkdir(parents=True, exist_ok=True)

        if not (target_dir / ".git").exists():
            # First time: full clone
            all_docs = []
            # Run normal clone
            self._shallow_clone_or_update(target_dir)
            # Full walk
            return self._full_walk_and_chunk(target_dir, "head"), [], []

        # Get current commit hash before pull
        old_commit = "head"
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(target_dir), capture_output=True, text=True)
            if res.returncode == 0:
                old_commit = res.stdout.strip()
        except Exception:
            pass

        # Pull updates
        try:
            subprocess.run(["git", "pull", "--depth", "1"], cwd=str(target_dir), capture_output=True, text=True, timeout=60)
        except Exception:
            pass

        # Get new commit hash after pull
        new_commit = old_commit
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(target_dir), capture_output=True, text=True)
            if res.returncode == 0:
                new_commit = res.stdout.strip()
        except Exception:
            pass

        # If commits are identical, no git changes occurred
        if old_commit == new_commit and old_commit != "head":
            return [], [], []

        # Determine git file changes via git diff
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
                    elif status.startswith("A"):
                        added_files.append(file_name)
                    elif status.startswith("D"):
                        deleted_files.append(file_name)
                    elif status.startswith("R") and len(parts) > 2:
                        deleted_files.append(parts[1])
                        added_files.append(parts[2])
        except Exception as e:
            print(f"Warning: git diff failed ({e}), falling back to full chunking with hash dedup.")
            return self._full_walk_and_chunk(target_dir, new_commit), [], []

        # 1. Purge stale chunks from DuckDB for modified and deleted files
        if store:
            for mod_file in modified_files + deleted_files:
                store.delete_documents_by_file(mod_file, source_prefix=f"{self.owner}/{self.repo}")

        # 2. Chunk ONLY modified and added files
        files_to_chunk = modified_files + added_files
        file_tasks: List[Tuple[str, str, str, str, str]] = []
        for rel_file in files_to_chunk:
            abs_file = target_dir / rel_file
            if abs_file.exists():
                file_tasks.append(
                    (str(abs_file), rel_file, f"https://github.com/{self.owner}/{self.repo}", self.branch, new_commit)
                )

        chunk_lists = parallel(_read_and_chunk_file_worker, file_tasks, n_workers=4)
        new_docs: List[Document] = [doc for sublist in chunk_lists for doc in sublist]

        return new_docs, modified_files, deleted_files

    def _full_walk_and_chunk(self, target_dir: Path, commit_hash: str) -> List[Document]:
        file_tasks: List[Tuple[str, str, str, str, str]] = []
        for root, dirs, files in os.walk(target_dir):
            rel_root = os.path.relpath(root, target_dir)
            if is_ignored_path(rel_root):
                continue
            for fname in files:
                rel_file = os.path.normpath(os.path.join(rel_root, fname))
                if is_ignored_path(rel_file):
                    continue
                abs_file = Path(root) / fname
                file_tasks.append(
                    (str(abs_file), rel_file, f"https://github.com/{self.owner}/{self.repo}", self.branch, commit_hash)
                )

        if self.max_files:
            file_tasks = file_tasks[: self.max_files]

        chunk_lists = parallel(_read_and_chunk_file_worker, file_tasks, n_workers=4)
        return [doc for sublist in chunk_lists for doc in sublist]

    async def load(self) -> List[Document]:
        """Executes shallow clone, file discovery, parallel AST chunking, and issue fetching."""
        target_dir = REPOS_CACHE_DIR / f"{self.owner}_{self.repo}"
        _, commit_hash = self._shallow_clone_or_update(target_dir)

        all_documents = self._full_walk_and_chunk(target_dir, commit_hash)

        # Optionally load GitHub issues
        if self.include_issues:
            issue_docs = await self._load_issues()
            all_documents.extend(issue_docs)

        return all_documents
