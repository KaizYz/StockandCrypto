from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.notes.app import create_app, db


class NotesAuthApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp_dir.name) / "notes_test.db"
        os.environ["NOTES_DB_PATH"] = str(db_path)
        os.environ["NOTES_AUTH_SECRET"] = "unit-test-secret"
        os.environ["NOTES_TOKEN_TTL_SECONDS"] = "3600"

        self.app = create_app()
        self.client = self.app.test_client()

        with self.app.app_context():
            db.drop_all()
            db.create_all()

    def tearDown(self) -> None:
        with self.app.app_context():
            db.session.remove()
            db.engine.dispose()
        self._tmp_dir.cleanup()
        os.environ.pop("NOTES_DB_PATH", None)
        os.environ.pop("NOTES_AUTH_SECRET", None)
        os.environ.pop("NOTES_TOKEN_TTL_SECONDS", None)

    def _register_and_login(self, username: str, email: str, password: str) -> str:
        self.client.post(
            "/api/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password,
            },
        )
        login_resp = self.client.post(
            "/api/auth/login",
            json={
                "email": email,
                "password": password,
            },
        )
        payload = login_resp.get_json() or {}
        return str(payload.get("token") or "")

    def test_register_and_login(self) -> None:
        register_resp = self.client.post(
            "/api/auth/register",
            json={
                "username": "test_user",
                "email": "test_user@example.com",
                "password": "StrongPass123",
            },
        )
        self.assertEqual(register_resp.status_code, 201)
        register_payload = register_resp.get_json() or {}
        self.assertTrue(register_payload.get("ok"))
        self.assertEqual(register_payload.get("user", {}).get("username"), "test_user")

        login_resp = self.client.post(
            "/api/auth/login",
            json={
                "email": "test_user@example.com",
                "password": "StrongPass123",
            },
        )
        self.assertEqual(login_resp.status_code, 200)
        login_payload = login_resp.get_json() or {}
        self.assertTrue(login_payload.get("ok"))
        self.assertTrue(login_payload.get("token"))

    def test_login_invalid_password(self) -> None:
        self.client.post(
            "/api/auth/register",
            json={
                "username": "alpha_user",
                "email": "alpha_user@example.com",
                "password": "StrongPass123",
            },
        )
        resp = self.client.post(
            "/api/auth/login",
            json={
                "email": "alpha_user@example.com",
                "password": "bad-pass",
            },
        )
        self.assertEqual(resp.status_code, 401)
        payload = resp.get_json() or {}
        self.assertFalse(payload.get("ok"))
        self.assertEqual(payload.get("error"), "invalid_credentials")

    def test_create_note_requires_auth(self) -> None:
        resp = self.client.post(
            "/api/notes",
            json={
                "title": "No Auth",
                "content": "Should fail",
            },
        )
        self.assertEqual(resp.status_code, 401)
        payload = resp.get_json() or {}
        self.assertEqual(payload.get("error"), "authorization_required")

    def test_create_list_and_public_notes(self) -> None:
        token = self._register_and_login("note_user", "note_user@example.com", "StrongPass123")
        self.assertTrue(token)
        headers = {"Authorization": f"Bearer {token}"}

        create_resp = self.client.post(
            "/api/notes",
            headers=headers,
            json={
                "title": "My first note",
                "content": "alpha content",
                "tags": "alpha,beta,alpha",
                "note_type": "note",
                "is_public": True,
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        create_payload = create_resp.get_json() or {}
        self.assertTrue(create_payload.get("ok"))
        created_item = create_payload.get("item", {})
        self.assertEqual(created_item.get("title"), "My first note")
        self.assertEqual(created_item.get("tags"), ["alpha", "beta"])

        mine_resp = self.client.get(
            "/api/notes",
            headers=headers,
            query_string={"mine": "true", "q": "alpha", "page_size": 20},
        )
        self.assertEqual(mine_resp.status_code, 200)
        mine_payload = mine_resp.get_json() or {}
        self.assertTrue(mine_payload.get("ok"))
        mine_items = mine_payload.get("items", [])
        self.assertEqual(len(mine_items), 1)
        self.assertEqual(mine_items[0].get("title"), "My first note")

        public_resp = self.client.get("/api/notes/public", query_string={"page_size": 10})
        self.assertEqual(public_resp.status_code, 200)
        public_payload = public_resp.get_json() or {}
        self.assertTrue(public_payload.get("ok"))
        public_items = public_payload.get("items", [])
        self.assertEqual(len(public_items), 1)
        author = public_items[0].get("author", {})
        self.assertEqual(author.get("username"), "note_user")


if __name__ == "__main__":
    unittest.main()
