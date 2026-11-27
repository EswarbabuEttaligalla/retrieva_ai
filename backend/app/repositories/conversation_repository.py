"""Owner-scoped repository for Supabase public.conversations records."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import Depends
from pydantic import BaseModel, ConfigDict

from backend.app.core.supabase_client import (
    SupabaseClient,
    SupabaseForbiddenError,
    SupabaseValidationError,
    get_supabase_client,
)


class SupabaseConversationRecord(BaseModel):
    """Domain model mirroring a row in Supabase public.conversations."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    owner_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class SupabaseConversationRepository:
    """Owner-scoped repository for public.conversations via PostgREST + caller RLS."""

    def __init__(self, client: SupabaseClient) -> None:
        self._client = client

    def _verify_owner_context(self, owner_id: UUID) -> None:
        if owner_id != self._client.owner_id:
            raise SupabaseForbiddenError(
                "Requested owner context does not match authenticated caller.",
                status_code=403,
                code="owner_mismatch",
            )

    async def create_conversation(
        self,
        owner_id: UUID,
        title: Optional[str] = None,
    ) -> SupabaseConversationRecord:
        """Create a new conversation owned by owner_id."""
        self._verify_owner_context(owner_id)
        clean_title = (title or "New Conversation").strip() or "New Conversation"
        if len(clean_title) > 200:
            clean_title = clean_title[:200]

        rows = await self._client.insert(
            "conversations",
            payload={
                "owner_id": str(owner_id),
                "title": clean_title,
            },
        )
        if not rows:
            raise SupabaseValidationError("Conversation creation returned no record.", status_code=500)
        return SupabaseConversationRecord.model_validate(rows[0])

    async def list_conversations(
        self,
        owner_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> List[SupabaseConversationRecord]:
        """List conversations owned by owner_id ordered by updated_at descending."""
        self._verify_owner_context(owner_id)
        rows = await self._client.select(
            "conversations",
            params={
                "owner_id": f"eq.{owner_id}",
                "order": "updated_at.desc",
                "limit": str(limit),
                "offset": str(offset),
            },
        )
        return [SupabaseConversationRecord.model_validate(row) for row in rows]

    async def get_conversation(
        self,
        conversation_id: UUID,
        owner_id: UUID,
    ) -> Optional[SupabaseConversationRecord]:
        """Fetch a conversation if owned by owner_id; returns None if missing or foreign."""
        self._verify_owner_context(owner_id)
        rows = await self._client.select(
            "conversations",
            params={
                "id": f"eq.{conversation_id}",
                "owner_id": f"eq.{owner_id}",
                "limit": "1",
            },
        )
        if not rows:
            return None
        return SupabaseConversationRecord.model_validate(rows[0])

    async def update_conversation_title(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        title: str,
    ) -> Optional[SupabaseConversationRecord]:
        """Rename an owned conversation; returns None if missing or owned by another user."""
        self._verify_owner_context(owner_id)
        clean_title = (title or "").strip()
        if not clean_title or len(clean_title) > 200:
            raise SupabaseValidationError("Conversation title must be 1-200 characters.", status_code=400)

        rows = await self._client.update(
            "conversations",
            params={
                "id": f"eq.{conversation_id}",
                "owner_id": f"eq.{owner_id}",
            },
            payload={"title": clean_title},
        )
        if not rows:
            return None
        return SupabaseConversationRecord.model_validate(rows[0])

    async def touch_conversation(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        title: Optional[str] = None,
    ) -> Optional[SupabaseConversationRecord]:
        """Touch an owned conversation's updated_at timestamp via PostgreSQL BEFORE UPDATE trigger.

        Uses the caller-permitted `update (title)` grant so `conversations_touch_updated_at`
        advances `updated_at` cleanly under existing Phase 1 RLS without extra column grants.
        """
        self._verify_owner_context(owner_id)
        resolved_title = (title or "").strip() if title is not None else ""
        if not resolved_title:
            existing = await self.get_conversation(conversation_id, owner_id)
            if existing is None:
                return None
            resolved_title = existing.title
        if len(resolved_title) > 200:
            resolved_title = resolved_title[:200]

        rows = await self._client.update(
            "conversations",
            params={
                "id": f"eq.{conversation_id}",
                "owner_id": f"eq.{owner_id}",
            },
            payload={"title": resolved_title},
        )
        if not rows:
            return None
        return SupabaseConversationRecord.model_validate(rows[0])

    async def delete_conversation(
        self,
        conversation_id: UUID,
        owner_id: UUID,
    ) -> bool:
        """Delete an owned conversation; returns False if missing or owned by another user."""
        self._verify_owner_context(owner_id)
        rows = await self._client.delete(
            "conversations",
            params={
                "id": f"eq.{conversation_id}",
                "owner_id": f"eq.{owner_id}",
            },
        )
        return len(rows) > 0


def get_supabase_conversation_repository(
    client: SupabaseClient = Depends(get_supabase_client),
) -> SupabaseConversationRepository:
    """FastAPI dependency providing a request-scoped SupabaseConversationRepository."""
    return SupabaseConversationRepository(client=client)
