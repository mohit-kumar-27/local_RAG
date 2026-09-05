"""
Integration tests for data ingestion loaders and error resilience:
- URL parsing for GitHub, ADO, Confluence
- Markdown conversion and HTML stripping
- Binary file detection
- Background job tracking
"""

import unittest
from pathlib import Path
import tempfile

from ingestion.github_loader import parse_github_url
from ingestion.ado_loader import parse_ado_url, html_to_plain_text
from ingestion.confluence_loader import parse_confluence_url
from rag.chunking import is_binary_file, is_ignored_path
from app.background import JOBS, create_job, get_active_ingestion_job, get_job


class TestIngestionLoaders(unittest.TestCase):

    def test_github_url_parsing(self):
        owner, repo, branch = parse_github_url("https://github.com/fastai/fastcore")
        self.assertEqual(owner, "fastai")
        self.assertEqual(repo, "fastcore")
        self.assertIsNone(branch)

        owner, repo, branch = parse_github_url("https://github.com/torvalds/linux.git")
        self.assertEqual(owner, "torvalds")
        self.assertEqual(repo, "linux")

        owner, repo, branch = parse_github_url("https://github.com/org/repo/tree/feature-branch")
        self.assertEqual(owner, "org")
        self.assertEqual(repo, "repo")
        self.assertEqual(branch, "feature-branch")

    def test_ado_url_parsing(self):
        info = parse_ado_url("https://dev.azure.com/myorg/myproject/_boards")
        self.assertEqual(info["organization"], "myorg")
        self.assertEqual(info["project"], "myproject")
        self.assertEqual(info["type"], "board")

        info_repo = parse_ado_url("https://dev.azure.com/myorg/myproject/_git/backend-service")
        self.assertEqual(info_repo["type"], "repo")
        self.assertEqual(info_repo["repo"], "backend-service")

    def test_confluence_url_parsing(self):
        info = parse_confluence_url("https://mycompany.atlassian.net/wiki/spaces/ENG/pages/987654321/Architecture")
        self.assertEqual(info["space_key"], "ENG")
        self.assertEqual(info["page_id"], "987654321")

        info_dc = parse_confluence_url("https://confluence.corp.local/pages/viewpage.action?pageId=12345")
        self.assertEqual(info_dc["page_id"], "12345")

    def test_html_stripping(self):
        raw_html = "<p>Critical bug: <b>Database timeout</b> occurred.</p><div>Check connection pool.</div>"
        cleaned = html_to_plain_text(raw_html)
        self.assertIn("Critical bug: Database timeout occurred.", cleaned)
        self.assertIn("Check connection pool.", cleaned)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<b>", cleaned)

    def test_binary_file_detection(self):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"MZ\x90\x00\x03")
            exe_path = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"print('hello world')")
            py_path = Path(f.name)

        try:
            self.assertTrue(is_binary_file(exe_path))
            self.assertFalse(is_binary_file(py_path))
        finally:
            exe_path.unlink(missing_ok=True)
            py_path.unlink(missing_ok=True)

    def test_ignored_path_detection(self):
        self.assertTrue(is_ignored_path(".git/HEAD"))
        self.assertTrue(is_ignored_path("node_modules/react/index.js"))
        self.assertTrue(is_ignored_path("src/__pycache__/app.cpython-312.pyc"))
        self.assertTrue(is_ignored_path(".env"))
        self.assertTrue(is_ignored_path("secrets/server.pem"))
        self.assertFalse(is_ignored_path("src/components/Button.tsx"))
        self.assertFalse(is_ignored_path("services/auth_service.py"))

    def test_background_job_lifecycle(self):
        job = create_job("github", "https://github.com/org/repo")
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.progress, 0)

        retrieved = get_job(job.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, job.id)

        job.add_log("Cloning complete")
        self.assertEqual(len(job.logs), 1)
        self.assertIn("Cloning complete", job.logs[0])

    def test_active_ingestion_job_tracking(self):
        JOBS.clear()
        self.assertIsNone(get_active_ingestion_job())

        job1 = create_job("github", "https://github.com/org/repo")
        self.assertEqual(get_active_ingestion_job().id, job1.id)

        job1.status = "running"
        self.assertEqual(get_active_ingestion_job().id, job1.id)

        job1.status = "completed"
        self.assertIsNone(get_active_ingestion_job())

    def test_ingestion_tab_lockout_ui(self):
        from app.ui_components import TabNavigation
        from fasthtml.common import to_xml

        # 1. Normal state: Ask Chatbot is clickable
        nav_normal = TabNavigation(active_tab="ingest", is_ingesting=False)
        xml_normal = to_xml(nav_normal)
        self.assertIn('hx-get="/tab/chat"', xml_normal)
        self.assertNotIn("Ask Chatbot (Locked)", xml_normal)

        # 2. Ingesting state: Ask Chatbot is locked
        nav_locked = TabNavigation(active_tab="ingest", is_ingesting=True)
        xml_locked = to_xml(nav_locked)
        self.assertNotIn('hx-get="/tab/chat"', xml_locked)
        self.assertIn("Ask Chatbot (Locked)", xml_locked)
        self.assertIn("Ingestion in progress (Chatbot locked)", xml_locked)
        self.assertIn("cursor-not-allowed", xml_locked)

    def test_tab_chat_route_blocked_during_ingestion(self):
        from starlette.testclient import TestClient
        from app.main import app

        JOBS.clear()
        job = create_job("github", "https://github.com/fastai/fastcore")
        job.status = "running"

        client = TestClient(app)
        resp = client.get("/tab/chat", headers={"HX-Request": "true"})
        self.assertEqual(resp.status_code, 200)
        # Verify it stays on Ingest Sources and displays warning alert
        self.assertIn("Ingestion is currently in progress", resp.text)
        self.assertIn("Ask Chatbot (Locked)", resp.text)

        # Reset job to completed
        job.status = "completed"
        resp_after = client.get("/tab/chat", headers={"HX-Request": "true"})
        self.assertEqual(resp_after.status_code, 200)
        self.assertNotIn("Ask Chatbot (Locked)", resp_after.text)


if __name__ == "__main__":
    unittest.main()
