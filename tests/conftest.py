import pytest
import sqlite3
import numpy as np
from pathlib import Path
import app.config as config
from app.database import init_db
from app.embeddings import embedding_service

@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Isolate database for each test session to avoid polluting data/campus_faq.db."""
    test_db_path = tmp_path / "test_faq.db"
    monkeypatch.setattr(config, "DB_PATH", test_db_path)
    init_db()
    yield test_db_path
