"""Service layer coordinating Supabase-backed conversation persistence, history, and query rewriting."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from fastapi import Depends

from backend.app.conversations.models import (
    ConversationDetail,
    ConversationRecord,
    MessageRecord,
)
from backend.app.conversations.query_rewriter import (
    ConversationalQueryRewriter,
    get_query_rewriter,
)
from backend.app.core.config import get_settings
from backend.app.repositories.conversation_repository import (
    SupabaseConversationRecord,
    SupabaseConversationRepository,
    get_supabase_conversation_repository,
)
from backend.app.repositories.message_repository import (
    SupabaseMessageRecord,
    SupabaseMessageRepository,
    get_message_repository,
)

logger = logging.getLogger("retrieva_ai.conversations.service")


class ConversationNotFoundError(Exception):
    """Raised when an operation targets a missing, malformed, or unowned conversation ID."""


def _normalize_uuid(value: Union[UUID, str]) -> Optional[UUID]:
    if isinstance(value, UUID):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def _to_conversation_record(
    rec: SupabaseConversationRecord,
    message_count: int = 0,
) -> ConversationRecord:
    return ConversationRecord(
        id=str(rec.id),
        title=rec.title,
        created_at=rec.created_at.isoformat(),
        updated_at=rec.updated_at.isoformat(),
        message_count=message_count,
    )


def _to_message_record(rec: SupabaseMessageRecord) -> MessageRecord:
    meta = dict(rec.metadata) if rec.metadata else {}
    if rec.citations and "citations" not in meta:
        meta["citations"] = list(rec.citations)
    return MessageRecord(
        id=str(rec.id),
        conversation_id=str(rec.conversation_id),
        role=rec.role,
        content=rec.content,
        raw_query=rec.raw_query,
        rewritten_query=rec.rewritten_query,
        created_at=rec.created_at.isoformat(),
        metadata=meta or None,
        citations=list(rec.citations),
    )


class ConversationService:
    """Owner-scoped, Supabase-backed conversation lifecycle, history, and query rewriting service."""

    def __init__(
        self,
        conversation_repo: SupabaseConversationRepository,
        message_repo: SupabaseMessageRepository,
        query_rewriter: Optional[ConversationalQueryRewriter] = None,
    ) -> None:
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.query_rewriter = query_rewriter or get_query_rewriter()

    @staticmethod
    def _require_owner_uuid(owner_id: Union[UUID, str]) -> UUID:
        parsed = _normalize_uuid(owner_id)
        if parsed is None:
            raise ValueError("A valid UUID owner_id is required for conversation operations.")
        return parsed

    async def create_conversation(
        self,
        owner_id: Union[UUID, str],
        title: Optional[str] = None,
    ) -> ConversationRecord:
        """Create a new persistent conversation owned by owner_id in Supabase."""
        owner_uuid = self._require_owner_uuid(owner_id)
        created = await self.conversation_repo.create_conversation(owner_id=owner_uuid, title=title)
        return _to_conversation_record(created, message_count=0)

    async def list_conversations(
        self,
        owner_id: Union[UUID, str],
        limit: int = 50,
        offset: int = 0,
    ) -> List[ConversationRecord]:
        """List persistent conversations belonging exclusively to owner_id."""
        owner_uuid = self._require_owner_uuid(owner_id)
        rows = await self.conversation_repo.list_conversations(
            owner_id=owner_uuid,
            limit=limit,
            offset=offset,
        )
        return [_to_conversation_record(row) for row in rows]

    async def get_conversation(
        self,
        conversation_id: str,
        owner_id: Union[UUID, str],
    ) -> Optional[ConversationRecord]:
        """Fetch metadata for an owned conversation; returns None if missing, invalid, or foreign."""
        owner_uuid = self._require_owner_uuid(owner_id)
        conv_uuid = _normalize_uuid(conversation_id)
        if conv_uuid is None:
            return None
        row = await self.conversation_repo.get_conversation(conversation_id=conv_uuid, owner_id=owner_uuid)
        if row is None:
            return None
        return _to_conversation_record(row)

    async def get_conversation_with_messages(
        self,
        conversation_id: str,
        owner_id: Union[UUID, str],
    ) -> Optional[ConversationDetail]:
        """Fetch an owned conversation and its full chronological message history."""
        owner_uuid = self._require_owner_uuid(owner_id)
        conv_uuid = _normalize_uuid(conversation_id)
        if conv_uuid is None:
            return None
        conv = await self.conversation_repo.get_conversation(conversation_id=conv_uuid, owner_id=owner_uuid)
        if conv is None:
            return None
        msg_rows = await self.message_repo.list_messages(conversation_id=conv_uuid, owner_id=owner_uuid)
        return ConversationDetail(
            id=str(conv.id),
            title=conv.title,
            created_at=conv.created_at.isoformat(),
            updated_at=conv.updated_at.isoformat(),
            messages=[_to_message_record(m) for m in msg_rows],
        )

    async def update_conversation_title(
        self,
        conversation_id: str,
        owner_id: Union[UUID, str],
        title: str,
    ) -> ConversationRecord:
        """Rename an owned conversation; raises ConversationNotFoundError if missing or foreign."""
        owner_uuid = self._require_owner_uuid(owner_id)
        conv_uuid = _normalize_uuid(conversation_id)
        if conv_uuid is None:
            raise ConversationNotFoundError("Conversation not found.")
        updated = await self.conversation_repo.update_conversation_title(
            conversation_id=conv_uuid,
            owner_id=owner_uuid,
            title=title,
        )
        if updated is None:
            raise ConversationNotFoundError("Conversation not found.")
        return _to_conversation_record(updated)

    async def delete_conversation(
        self,
        conversation_id: str,
        owner_id: Union[UUID, str],
    ) -> bool:
        """Delete an owned conversation; raises ConversationNotFoundError if missing or foreign."""
        owner_uuid = self._require_owner_uuid(owner_id)
        conv_uuid = _normalize_uuid(conversation_id)
        if conv_uuid is None:
            raise ConversationNotFoundError("Conversation not found.")
        deleted = await self.conversation_repo.delete_conversation(
            conversation_id=conv_uuid,
            owner_id=owner_uuid,
        )
        if not deleted:
            raise ConversationNotFoundError("Conversation not found.")
        return True

    @staticmethod
    def _generate_title_from_query(query: str) -> str:
        """Derive a clean, concise conversation title from the initial user query."""
        clean = " ".join(query.strip().split())
        if len(clean) > 50:
            truncated = clean[:50].rsplit(" ", 1)[0]
            return truncated if truncated else clean[:50]
        return clean or "New Conversation"

    async def prepare_conversational_turn(
        self,
        conversation_id: str,
        owner_id: Union[UUID, str],
        raw_query: str,
    ) -> Tuple[str, List[MessageRecord]]:
        """Verify ownership, load bounded recent history for this owner, and rewrite follow-up query."""
        owner_uuid = self._require_owner_uuid(owner_id)
        conv_uuid = _normalize_uuid(conversation_id)
        if conv_uuid is None:
            raise ConversationNotFoundError("Conversation not found.")

        conv = await self.conversation_repo.get_conversation(conversation_id=conv_uuid, owner_id=owner_uuid)
        if conv is None:
            raise ConversationNotFoundError("Conversation not found.")

        settings = get_settings()
        recent_rows = await self.message_repo.get_recent_history(
            conversation_id=conv_uuid,
            owner_id=owner_uuid,
            window=settings.CONVERSATION_HISTORY_WINDOW,
        )
        history = [_to_message_record(m) for m in recent_rows]

        rewritten_query = await self.query_rewriter.rewrite_async(
            raw_query=raw_query,
            history=history,
            window=settings.CONVERSATION_HISTORY_WINDOW,
        )
        return rewritten_query, history

    async def record_user_message(
        self,
        conversation_id: str,
        owner_id: Union[UUID, str],
        raw_query: str,
        rewritten_query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageRecord:
        """Persist an owned user turn, auto-title new conversations, and advance updated_at."""
        owner_uuid = self._require_owner_uuid(owner_id)
        conv_uuid = _normalize_uuid(conversation_id)
        if conv_uuid is None:
            raise ConversationNotFoundError("Conversation not found.")

        conv = await self.conversation_repo.get_conversation(conversation_id=conv_uuid, owner_id=owner_uuid)
        if conv is None:
            raise ConversationNotFoundError("Conversation not found.")

        inserted = await self.message_repo.insert_user_message(
            conversation_id=conv_uuid,
            owner_id=owner_uuid,
            content=raw_query,
            raw_query=raw_query,
            rewritten_query=rewritten_query,
            metadata=metadata,
        )

        # Auto-title if default placeholder, otherwise touch title to advance conversations.updated_at
        if conv.title == "New Conversation" or not conv.title.strip():
            target_title = self._generate_title_from_query(raw_query)
            await self.conversation_repo.update_conversation_title(
                conversation_id=conv_uuid,
                owner_id=owner_uuid,
                title=target_title,
            )
        else:
            await self.conversation_repo.touch_conversation(
                conversation_id=conv_uuid,
                owner_id=owner_uuid,
                title=conv.title,
            )

        return _to_message_record(inserted)

    async def record_assistant_message(
        self,
        conversation_id: str,
        owner_id: Union[UUID, str],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> MessageRecord:
        """Persist a verified assistant response via the owner-scoped Supabase RPC."""
        owner_uuid = self._require_owner_uuid(owner_id)
        conv_uuid = _normalize_uuid(conversation_id)
        if conv_uuid is None:
            raise ConversationNotFoundError("Conversation not found.")

        conv = await self.conversation_repo.get_conversation(conversation_id=conv_uuid, owner_id=owner_uuid)
        if conv is None:
            raise ConversationNotFoundError("Conversation not found.")

        meta_payload = dict(metadata) if metadata is not None else {}
        resolved_citations = list(citations) if citations is not None else list(meta_payload.get("citations") or [])
        if resolved_citations and "citations" not in meta_payload:
            meta_payload["citations"] = resolved_citations

        inserted = await self.message_repo.insert_assistant_message(
            conversation_id=conv_uuid,
            owner_id=owner_uuid,
            content=content,
            metadata=meta_payload,
            citations=resolved_citations,
        )
        return _to_message_record(inserted)


def get_conversation_service(
    conversation_repo: SupabaseConversationRepository = Depends(get_supabase_conversation_repository),
    message_repo: SupabaseMessageRepository = Depends(get_message_repository),
    query_rewriter: ConversationalQueryRewriter = Depends(get_query_rewriter),
) -> ConversationService:
    """FastAPI dependency constructing a request-scoped Supabase-backed ConversationService."""
    return ConversationService(
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        query_rewriter=query_rewriter,
    )
