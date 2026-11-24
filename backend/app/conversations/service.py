"""Service layer coordinating conversation persistence, history management, and query rewriting."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.app.conversations.models import (
    ConversationDetail,
    ConversationRecord,
    MessageRecord,
)
from backend.app.conversations.query_rewriter import (
    ConversationalQueryRewriter,
    get_query_rewriter,
)
from backend.app.conversations.repository import (
    ConversationRepository,
    get_conversation_repository,
)
from backend.app.core.config import get_settings

logger = logging.getLogger("retrieva_ai.conversations.service")


class ConversationNotFoundError(Exception):
    """Raised when an operation targets an uncataloged conversation ID."""
    pass


class ConversationService:
    """Coordinates conversation lifecycle, multi-turn message storage, and contextual query rewriting."""

    def __init__(
        self,
        repository: Optional[ConversationRepository] = None,
        query_rewriter: Optional[ConversationalQueryRewriter] = None,
    ):
        self.repository = repository or get_conversation_repository()
        self.query_rewriter = query_rewriter or get_query_rewriter()

    def create_conversation(self, title: Optional[str] = None) -> ConversationRecord:
        """Create a new conversation session."""
        return self.repository.create_conversation(title=title)

    async def create_conversation_async(self, title: Optional[str] = None) -> ConversationRecord:
        """Create a new conversation session asynchronously."""
        return await asyncio.to_thread(self.create_conversation, title)

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[ConversationRecord]:
        """List persistent conversations ordered by most recently updated."""
        return self.repository.list_conversations(limit=limit, offset=offset)

    async def list_conversations_async(self, limit: int = 50, offset: int = 0) -> List[ConversationRecord]:
        """List persistent conversations asynchronously."""
        return await asyncio.to_thread(self.list_conversations, limit, offset)

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        """Fetch metadata for a single conversation."""
        return self.repository.get_conversation(conversation_id)

    async def get_conversation_async(self, conversation_id: str) -> Optional[ConversationRecord]:
        """Fetch metadata for a single conversation asynchronously."""
        return await asyncio.to_thread(self.get_conversation, conversation_id)

    def get_conversation_with_messages(self, conversation_id: str) -> Optional[ConversationDetail]:
        """Fetch conversation details including full chronological message history."""
        conv = self.repository.get_conversation(conversation_id)
        if not conv:
            return None
        messages = self.repository.get_messages(conversation_id)
        return ConversationDetail(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            messages=messages,
        )

    async def get_conversation_with_messages_async(self, conversation_id: str) -> Optional[ConversationDetail]:
        """Fetch conversation details including full chronological message history asynchronously."""
        return await asyncio.to_thread(self.get_conversation_with_messages, conversation_id)

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        conv = self.repository.get_conversation(conversation_id)
        if not conv:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found.")
        return self.repository.update_conversation(conversation_id, title=title)

    async def update_conversation_title_async(self, conversation_id: str, title: str) -> bool:
        """Update conversation title asynchronously."""
        return await asyncio.to_thread(self.update_conversation_title, conversation_id, title)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and its associated messages."""
        deleted = self.repository.delete_conversation(conversation_id)
        if not deleted:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found.")
        return True

    async def delete_conversation_async(self, conversation_id: str) -> bool:
        """Delete conversation and its associated messages asynchronously."""
        return await asyncio.to_thread(self.delete_conversation, conversation_id)

    def _generate_title_from_query(self, query: str) -> str:
        """Derive a clean, concise conversation title from the initial user query."""
        clean = " ".join(query.strip().split())
        if len(clean) > 50:
            truncated = clean[:50].rsplit(" ", 1)[0]
            return truncated if truncated else clean[:50]
        return clean or "New Conversation"

    def record_user_message(
        self,
        conversation_id: str,
        raw_query: str,
        rewritten_query: str,
    ) -> MessageRecord:
        """Persist a user turn and automatically derive title if conversation is new."""
        conv = self.repository.get_conversation(conversation_id)
        if not conv:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found.")

        # If the conversation has a default placeholder title, derive title from first question
        if conv.title == "New Conversation" or not conv.title.strip():
            derived_title = self._generate_title_from_query(raw_query)
            self.repository.update_conversation(conversation_id, title=derived_title)

        return self.repository.add_message(
            conversation_id=conversation_id,
            role="user",
            content=raw_query,
            raw_query=raw_query,
            rewritten_query=rewritten_query,
        )

    async def record_user_message_async(
        self,
        conversation_id: str,
        raw_query: str,
        rewritten_query: str,
    ) -> MessageRecord:
        """Persist a user turn asynchronously in a worker thread."""
        return await asyncio.to_thread(
            self.record_user_message,
            conversation_id,
            raw_query,
            rewritten_query,
        )

    def record_assistant_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageRecord:
        """Persist a verified assistant response turn with citation and diagnostic metadata."""
        conv = self.repository.get_conversation(conversation_id)
        if not conv:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found.")

        return self.repository.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            metadata=metadata,
        )

    async def record_assistant_message_async(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageRecord:
        """Persist a verified assistant response turn asynchronously in a worker thread."""
        return await asyncio.to_thread(
            self.record_assistant_message,
            conversation_id,
            content,
            metadata,
        )

    async def prepare_conversational_turn(
        self,
        conversation_id: str,
        raw_query: str,
    ) -> Tuple[str, List[MessageRecord]]:
        """Validate conversation existence, retrieve history, and rewrite follow-up query asynchronously."""
        conv = await self.get_conversation_async(conversation_id)
        if not conv:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found.")

        settings = get_settings()
        history = await asyncio.to_thread(
            self.repository.get_recent_history,
            conversation_id,
            settings.CONVERSATION_HISTORY_WINDOW,
        )

        rewritten_query = await self.query_rewriter.rewrite_async(
            raw_query=raw_query,
            history=history,
            window=settings.CONVERSATION_HISTORY_WINDOW,
        )

        return rewritten_query, history

    def prepare_conversational_turn_sync(
        self,
        conversation_id: str,
        raw_query: str,
    ) -> Tuple[str, List[MessageRecord]]:
        """Validate conversation existence, retrieve history, and rewrite follow-up query synchronously."""
        conv = self.repository.get_conversation(conversation_id)
        if not conv:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found.")

        settings = get_settings()
        history = self.repository.get_recent_history(
            conversation_id,
            window=settings.CONVERSATION_HISTORY_WINDOW,
        )

        rewritten_query = self.query_rewriter.rewrite(
            raw_query=raw_query,
            history=history,
            window=settings.CONVERSATION_HISTORY_WINDOW,
        )

        return rewritten_query, history


_service_instance: Optional[ConversationService] = None


def get_conversation_service() -> ConversationService:
    """Return singleton instance of ConversationService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ConversationService()
    return _service_instance
