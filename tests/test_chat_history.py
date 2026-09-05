"""
Unit tests for DuckDB persistent chat history management:
- Chat session CRUD & ordering
- Message CRUD & JSON citations
- Sliding-window context retrieval
- Message edit & subsequent turn pruning (delete_messages_after)
- Message pair deletion (user + assistant)
- Chat cascade deletion
"""

import tempfile
import unittest
from pathlib import Path

from rag.duckdb_store import DuckDBStore


class TestChatHistory(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_chat.duckdb"
        self.store = DuckDBStore(db_path=self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_create_and_list_chats(self):
        c1 = self.store.create_chat("chat-1", "First Chat")
        c2 = self.store.create_chat("chat-2", "Second Chat")

        self.assertEqual(c1.id, "chat-1")
        self.assertEqual(c1.title, "First Chat")

        chats = self.store.list_chats()
        self.assertEqual(len(chats), 2)
        self.assertEqual(chats[0].id, "chat-2")
        self.assertEqual(chats[1].id, "chat-1")

        self.store.update_chat_title("chat-1", "Updated First Chat")
        updated = self.store.get_chat("chat-1")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.title, "Updated First Chat")

    def test_add_and_retrieve_messages_with_citations(self):
        self.store.create_chat("chat-1", "Auth Questions")

        u_msg = self.store.add_chat_message(
            message_id="msg-1",
            chat_id="chat-1",
            role="user",
            content="How does JWT authentication work?",
        )
        self.assertEqual(u_msg.role, "user")
        self.assertIsNone(u_msg.citations)

        citations = [
            {"file_path": "src/auth.py", "doc_type": "code", "score": 0.92, "content": "def verify_jwt(): pass"},
            {"work_item_id": "1042", "doc_type": "ticket", "score": 0.85, "content": "Bug in JWT validation"},
        ]
        a_msg = self.store.add_chat_message(
            message_id="msg-2",
            chat_id="chat-1",
            role="assistant",
            content="JWT tokens are verified using the secret key.",
            citations=citations,
        )
        self.assertEqual(a_msg.role, "assistant")
        self.assertIsNotNone(a_msg.citations)
        self.assertEqual(len(a_msg.citations), 2)
        self.assertEqual(a_msg.citations[0]["file_path"], "src/auth.py")

        all_msgs = self.store.get_chat_messages("chat-1")
        self.assertEqual(len(all_msgs), 2)
        self.assertEqual(all_msgs[0].id, "msg-1")
        self.assertEqual(all_msgs[1].id, "msg-2")
        self.assertEqual(all_msgs[1].citations[1]["work_item_id"], "1042")

    def test_sliding_window_limit(self):
        self.store.create_chat("chat-1", "Long Conversation")

        for i in range(1, 16):
            role = "user" if i % 2 != 0 else "assistant"
            self.store.add_chat_message(
                message_id=f"msg-{i:02d}",
                chat_id="chat-1",
                role=role,
                content=f"Message content {i}",
            )

        window = self.store.get_chat_messages("chat-1", limit=10)
        self.assertEqual(len(window), 10)
        self.assertEqual(window[0].id, "msg-06")
        self.assertEqual(window[-1].id, "msg-15")

    def test_update_message_content(self):
        self.store.create_chat("chat-1", "Editing Chat")
        self.store.add_chat_message("msg-1", "chat-1", "user", "Original question")

        self.store.update_message_content("msg-1", "Edited question")
        msg = self.store.get_message("msg-1")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.content, "Edited question")

    def test_delete_message_pair(self):
        self.store.create_chat("chat-1", "Pair Deletion")
        self.store.add_chat_message("u1", "chat-1", "user", "Q1")
        self.store.add_chat_message("a1", "chat-1", "assistant", "A1")
        self.store.add_chat_message("u2", "chat-1", "user", "Q2")
        self.store.add_chat_message("a2", "chat-1", "assistant", "A2")

        deleted = self.store.delete_message_pair("chat-1", "u1")
        self.assertTrue(deleted)

        remaining = self.store.get_chat_messages("chat-1")
        self.assertEqual(len(remaining), 2)
        self.assertEqual(remaining[0].id, "u2")
        self.assertEqual(remaining[1].id, "a2")

    def test_delete_messages_after(self):
        self.store.create_chat("chat-1", "Turn Pruning")
        self.store.add_chat_message("u1", "chat-1", "user", "Q1")
        self.store.add_chat_message("a1", "chat-1", "assistant", "A1")
        self.store.add_chat_message("u2", "chat-1", "user", "Q2")
        self.store.add_chat_message("a2", "chat-1", "assistant", "A2")

        pruned = self.store.delete_messages_after("chat-1", "u1")
        self.assertEqual(pruned, 1)

        remaining = self.store.get_chat_messages("chat-1")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].id, "u1")

    def test_delete_chat_cascade(self):
        self.store.create_chat("chat-1", "To Be Deleted")
        self.store.add_chat_message("u1", "chat-1", "user", "Q1")
        self.store.add_chat_message("a1", "chat-1", "assistant", "A1")

        success = self.store.delete_chat("chat-1")
        self.assertTrue(success)

        self.assertIsNone(self.store.get_chat("chat-1"))
        self.assertEqual(len(self.store.get_chat_messages("chat-1")), 0)
        self.assertEqual(len(self.store.list_chats()), 0)

    def test_chat_sidebar_kwargs_and_oob(self):
        from app.ui_components import ChatSidebar
        chats = [self.store.create_chat("chat-1", "Test Chat")]
        sidebar = ChatSidebar(chats=chats, active_chat_id="chat-1", hx_swap_oob="true")
        from fasthtml.common import to_xml
        rendered = to_xml(sidebar)
        self.assertIn('hx-swap-oob="true"', rendered)
        self.assertIn('id="chat-sidebar"', rendered)

    def test_post_chat_and_session_routes(self):
        from starlette.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        # 1. Test POST /api/chat (should return turn element, sidebar OOB, and active-chat-id-input OOB)
        resp = client.post("/api/chat", data={"query": "How does testing work?"}, headers={"HX-Request": "true"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("turn-", resp.text)
        self.assertIn('id="chat-sidebar"', resp.text)
        self.assertIn('hx-swap-oob="true"', resp.text)
        self.assertIn('id="active-chat-id-input"', resp.text)

        # 2. Test GET /api/chats/new
        resp_new = client.get("/api/chats/new", headers={"HX-Request": "true"})
        self.assertEqual(resp_new.status_code, 200)
        self.assertIn('id="chat-sidebar"', resp_new.text)
        self.assertIn('id="chat-main-area"', resp_new.text)

        # 3. Test GET /api/chats/{chat_id}
        from app.main import store
        chats = store.list_chats()
        self.assertTrue(len(chats) > 0)
        first_chat = chats[0]
        resp_session = client.get(f"/api/chats/{first_chat.id}", headers={"HX-Request": "true"})
        self.assertEqual(resp_session.status_code, 200)
        self.assertIn('id="chat-sidebar"', resp_session.text)
        self.assertIn('id="chat-main-area"', resp_session.text)

    def test_scrollbars_and_tab_swapping(self):
        from starlette.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        # 1. Ingest page should have overflow-y-scroll and scrollbar CSS
        r_ingest = client.get("/?tab=ingest")
        self.assertEqual(r_ingest.status_code, 200)
        self.assertIn("overflow-y-scroll", r_ingest.text)
        self.assertIn("::-webkit-scrollbar", r_ingest.text)
        self.assertIn("scrollbar-gutter: stable", r_ingest.text)

        # 2. Chat page should have overflow-y-scroll on chat-main-area
        r_chat = client.get("/?tab=chat")
        self.assertEqual(r_chat.status_code, 200)
        self.assertIn("overflow-y-scroll", r_chat.text)

        # 3. Swapping to Ingest tab via HTMX must return tab-content with overflow-y-scroll
        r_htmx_ingest = client.get("/tab/ingest", headers={"HX-Request": "true"})
        self.assertEqual(r_htmx_ingest.status_code, 200)
        self.assertIn('id="tab-content"', r_htmx_ingest.text)
        self.assertIn("overflow-y-scroll", r_htmx_ingest.text)

        # 4. Swapping to Chat tab via HTMX must return tab-content and chat-main-area with overflow-y-scroll
        r_htmx_chat = client.get("/tab/chat", headers={"HX-Request": "true"})
        self.assertEqual(r_htmx_chat.status_code, 200)
        self.assertIn('id="tab-content"', r_htmx_chat.text)
        self.assertIn('id="chat-main-area"', r_htmx_chat.text)
        self.assertIn("overflow-y-scroll", r_htmx_chat.text)

    def test_assistant_message_editing(self):
        import uuid
        from starlette.testclient import TestClient
        from app.main import app, store

        client = TestClient(app)

        test_chat_id = f"chat-{uuid.uuid4().hex[:8]}"
        user_msg_id = f"u-{uuid.uuid4().hex[:8]}"
        asst_msg_id = f"a-{uuid.uuid4().hex[:8]}"

        # 1. Setup chat with user and assistant messages
        chat = store.create_chat(test_chat_id, "Edit Assistant Test")
        store.add_chat_message(user_msg_id, chat.id, "user", "What is FastAPI?")
        citations = [
            {"file_path": "app/main.py", "doc_type": "code", "score": 0.88, "content": "app = FastHTML()"}
        ]
        store.add_chat_message(asst_msg_id, chat.id, "assistant", "FastAPI is a Python web framework.", citations=citations)

        # 2. Check AssistantMessageBubble renders the Edit button
        resp_view = client.get(f"/api/chats/{chat.id}", headers={"HX-Request": "true"})
        self.assertEqual(resp_view.status_code, 200)
        self.assertIn(f'id="assistant-bubble-{asst_msg_id}"', resp_view.text)
        self.assertIn(f'/api/chats/{chat.id}/assistant-messages/{asst_msg_id}/edit-form', resp_view.text)
        self.assertIn('Edit', resp_view.text)

        # 3. GET /edit-form returns the inline edit form
        resp_form = client.get(f"/api/chats/{chat.id}/assistant-messages/{asst_msg_id}/edit-form", headers={"HX-Request": "true"})
        self.assertEqual(resp_form.status_code, 200)
        self.assertIn(f'id="assistant-bubble-{asst_msg_id}"', resp_form.text)
        self.assertIn('name="content"', resp_form.text)
        self.assertIn("FastAPI is a Python web framework.", resp_form.text)
        self.assertIn(f'/api/chats/{chat.id}/assistant-messages/{asst_msg_id}/edit', resp_form.text)
        self.assertIn(f'/api/chats/{chat.id}/assistant-messages/{asst_msg_id}/cancel-edit', resp_form.text)

        # 4. GET /cancel-edit restores original bubble
        resp_cancel = client.get(f"/api/chats/{chat.id}/assistant-messages/{asst_msg_id}/cancel-edit", headers={"HX-Request": "true"})
        self.assertEqual(resp_cancel.status_code, 200)
        self.assertIn(f'id="assistant-bubble-{asst_msg_id}"', resp_cancel.text)
        self.assertIn("FastAPI is a Python web framework.", resp_cancel.text)
        self.assertNotIn('name="content"', resp_cancel.text)

        # 5. POST /edit saves modified content to DuckDB and re-renders bubble
        new_answer = "FastAPI is a high-performance Python framework built on Starlette and Pydantic."
        resp_save = client.post(
            f"/api/chats/{chat.id}/assistant-messages/{asst_msg_id}/edit",
            data={"content": new_answer},
            headers={"HX-Request": "true"},
        )
        self.assertEqual(resp_save.status_code, 200)
        self.assertIn(f'id="assistant-bubble-{asst_msg_id}"', resp_save.text)
        self.assertIn("high-performance Python framework", resp_save.text)

        # 6. Verify DuckDB reflects the edited content and citations remain intact
        updated_msg = store.get_message(asst_msg_id)
        self.assertIsNotNone(updated_msg)
        self.assertEqual(updated_msg.content, new_answer)
        self.assertIsNotNone(updated_msg.citations)
        self.assertEqual(len(updated_msg.citations), 1)
        self.assertEqual(updated_msg.citations[0]["file_path"], "app/main.py")

    def test_collapsible_sidebar(self):
        from starlette.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.get("/?tab=chat")
        self.assertEqual(resp.status_code, 200)

        # 1. Verify ChatSidebar contains the collapse toggle button with toggleChatSidebar()
        self.assertIn("toggleChatSidebar()", resp.text)
        self.assertIn("Collapse sidebar", resp.text)

        # 2. Verify ChatMainArea contains the expand button with id="chat-sidebar-expand-btn"
        self.assertIn('id="chat-sidebar-expand-btn"', resp.text)

        # 3. Verify CSS rules for smooth collapsible transition and collapsed state exist
        self.assertIn("#chat-sidebar.collapsed", resp.text)
        self.assertIn("transition: width", resp.text)

        # 4. Verify JS functions for toggling, localStorage persistence, and shortcut exist
        self.assertIn("function toggleChatSidebar()", resp.text)
        self.assertIn("function initChatSidebarState()", resp.text)
        self.assertIn("chat_sidebar_collapsed", resp.text)


if __name__ == "__main__":
    unittest.main()


