"""Repository for persistent SQLite storage of conversations and message history."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.app.conversations.database import get_db_connection
from backend.app.conversations.models import ConversationRecord, MessageRecord

logger = logging.getLogger("retrieva_ai.conversations.repository")


class ConversationRepository:
    """Data Access Object managing persistence for conversations and messages."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def _utc_now(self) -> str:
        """Generate current UTC ISO-8601 timestamp string."""
        return datetime.now(timezone.utc).isoformat()

    def create_conversation(
        self,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> ConversationRecord:
        """Create a new conversation record."""
        cid = conversation_id or str(uuid.uuid4())
        clean_title = (title or "New Conversation").strip()
        now = self._utc_now()

        with get_db_connection(self.db_path) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO conversations (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?);
                    """,
                    (cid, clean_title, now, now),
                )

        logger.debug("Created conversation '%s' with title '%s'", cid, clean_title)
        return ConversationRecord(
            id=cid,
            title=clean_title,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        """Retrieve conversation by ID, including total message count."""
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.id = ?
                GROUP BY c.id;
                """,
                (conversation_id,),
            ).fetchone()

            if not row:
                return None

            return ConversationRecord(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[ConversationRecord]:
        """List all conversations ordered by most recently updated."""
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?;
                """,
                (limit, offset),
            ).fetchall()

            return [
                ConversationRecord(
                    id=r["id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    message_count=r["message_count"],
                )
                for r in rows
            ]

    def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> bool:
        """Update conversation title and/or timestamp."""
        now = updated_at or self._utc_now()
        with get_db_connection(self.db_path) as conn:
            with conn:
                if title is not None:
                    cursor = conn.execute(
                        """
                        UPDATE conversations
                        SET title = ?, updated_at = ?
                        WHERE id = ?;
                        """,
                        (title.strip(), now, conversation_id),
                    )
                else:
                    cursor = conn.execute(
                        """
                        UPDATE conversations
                        SET updated_at = ?
                        WHERE id = ?;
                        """,
                        (now, conversation_id),
                    )
                return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and cascade associated messages."""
        with get_db_connection(self.db_path) as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = ?;",
                    (conversation_id,),
                )
                deleted = cursor.rowcount > 0

        if deleted:
            logger.info("Deleted conversation '%s' and cascaded messages.", conversation_id)
        return deleted

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        raw_query: Optional[str] = None,
        rewritten_query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> MessageRecord:
        """Insert a message into conversation history with transactional timestamp updates."""
        mid = message_id or str(uuid.uuid4())
        now = created_at or self._utc_now()
        meta_json = json.dumps(metadata) if metadata is not None else None

        with get_db_connection(self.db_path) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO messages (
                        id, conversation_id, role, content, raw_query, rewritten_query, created_at, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (mid, conversation_id, role, content, raw_query, rewritten_query, now, meta_json),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?;",
                    (now, conversation_id),
                )

        logger.debug("Added %s message '%s' to conversation '%s'", role, mid, conversation_id)
        return MessageRecord(
            id=mid,
            conversation_id=conversation_id,
            role=role,
            content=content,
            raw_query=raw_query,
            rewritten_query=rewritten_query,
            created_at=now,
            metadata=metadata,
        )

    def get_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[MessageRecord]:
        """Retrieve all messages for a conversation in chronological order (created_at ASC)."""
        query = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC"
        params: List[Any] = [conversation_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_message(r) for r in rows]

    def get_recent_history(self, conversation_id: str, window: int = 5) -> List[MessageRecord]:
        """Retrieve the most recent N message pairs (2 * window messages) in chronological order."""
        max_messages = max(1, window * 2)
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ) sub
                ORDER BY created_at ASC;
                """,
                (conversation_id, max_messages),
            ).fetchall()

            return [self._row_to_message(r) for r in rows]

    def _row_to_message(self, row: sqlite3.Row) -> MessageRecord:
        """Convert a sqlite3.Row to a domain MessageRecord."""
        raw_meta = row["metadata"]
        parsed_meta = None
        if raw_meta:
            try:
                parsed_meta = json.loads(raw_meta)
            except Exception as exc:
                logger.debug("Failed parsing message metadata JSON: %s", exc)

        return MessageRecord(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            raw_query=row["raw_query"],
            rewritten_query=row["rewritten_query"],
            created_at=row["created_at"],
            metadata=parsed_meta,
        )


_repository_instance: Optional[ConversationRepository] = None


def get_conversation_repository() -> ConversationRepository:
    """Return singleton instance of ConversationRepository."""
    global _repository_instance
    if _repository_instance is None:
        _repository_instance = ConversationRepository()
    return _repository_instance
