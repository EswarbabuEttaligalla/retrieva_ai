"""SQLite Database Connection and Schema Management for Conversations."""

import os
import sqlite3
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from backend.app.core.config import get_settings

logger = logging.getLogger("retrieva_ai.conversations.database")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    raw_query TEXT,
    rewritten_query TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
"""


@contextmanager
def get_db_connection(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Provide a contextual SQLite connection with foreign keys and WAL enabled.

    Each caller receives an isolated connection that is cleanly closed upon block exit.
    This guarantees thread isolation and prevents cross-thread connection sharing.
    """
    settings = get_settings()
    path = db_path or settings.CONVERSATION_DB_PATH

    # Ensure target directory exists
    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    conn = sqlite3.connect(
        path,
        timeout=30.0,
        check_same_thread=True,
    )
    conn.row_factory = sqlite3.Row

    try:
        # Enable foreign key constraint enforcement
        conn.execute("PRAGMA foreign_keys = ON;")
        # Enable Write-Ahead Logging for concurrency and crash-resilience
        conn.execute("PRAGMA journal_mode = WAL;")
        # Set busy timeout in milliseconds to prevent immediate locks during concurrent access
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        yield conn
    finally:
        try:
            conn.close()
        except Exception as exc:
            logger.debug("Notice closing SQLite connection: %s", exc)


def init_conversation_db(db_path: Optional[str] = None) -> None:
    """Initialize the conversation database and ensure all tables and indexes exist."""
    settings = get_settings()
    path = db_path or settings.CONVERSATION_DB_PATH
    logger.info("Initializing conversation database at '%s'...", path)

    with get_db_connection(path) as conn:
        with conn:
            conn.executescript(SCHEMA_SQL)

    logger.info("Conversation database schema initialized successfully.")
