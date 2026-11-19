"""Shared test setup."""

import os

# FastAPI's TestClient connects as "testclient" to host "testserver"; the server accepts those
# names only in tests (services/security.py).
os.environ["NOTETAKER_TESTING"] = "1"
