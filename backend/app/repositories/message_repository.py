"""Owner-scoped repository for Supabase public.messages records."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import Depends
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.rpc_auth import (
    build_backend_rpc_proof,
    compute_record_assistant_message_payload_digest,
    resolve_backend_rpc_hmac_secret,
)
from backend.app.core.supabase_client import (
    SupabaseClient,
    SupabaseForbiddenError,
    SupabaseValidationError,
    get_supabase_client,
)


class SupabaseMessageRecord(BaseModel):
    """Domain model mirroring a row in Supabase public.messages."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    conversation_id: UUID
    owner_id: UUID
    role: str
    content: str
    raw_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class SupabaseMessageRepository:
    """Owner-scoped repository for public.messages via PostgREST + caller RLS.

    Direct assistant-message insertion (`role='assistant'`) is intentionally
    disallowed by Phase 1 RLS policy `messages_insert_user_turn_in_owned_conversation`
    and requires an approved SECURITY DEFINER RPC before live assistant writes.
    """

    def __init__(self, client: SupabaseClient, rpc_hmac_secret: Optional[str] = None) -> None:
        self._client = client
        self._rpc_hmac_secret_override = rpc_hmac_secret

    def __repr__(self) -> str:
        return f"SupabaseMessageRepository(owner_id={self._client.owner_id})"

    def _verify_owner_context(self, owner_id: UUID) -> None:
        if owner_id != self._client.owner_id:
            raise SupabaseForbiddenError(
                "Requested owner context does not match authenticated caller.",
                status_code=403,
                code="owner_mismatch",
            )

    async def list_messages(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        limit: Optional[int] = None,
    ) -> List[SupabaseMessageRecord]:
        """List messages in an owned conversation ordered chronologically."""
        self._verify_owner_context(owner_id)
        params: Dict[str, Any] = {
            "conversation_id": f"eq.{conversation_id}",
            "owner_id": f"eq.{owner_id}",
            "order": "created_at.asc",
        }
        if limit is not None:
            params["limit"] = str(limit)

        rows = await self._client.select("messages", params=params)
        return [SupabaseMessageRecord.model_validate(row) for row in rows]

    async def get_recent_history(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        window: int = 5,
    ) -> List[SupabaseMessageRecord]:
        """Fetch up to `window` recent user/assistant turn pairs in chronological order."""
        self._verify_owner_context(owner_id)
        if window <= 0:
            return []

        rows = await self._client.select(
            "messages",
            params={
                "conversation_id": f"eq.{conversation_id}",
                "owner_id": f"eq.{owner_id}",
                "order": "created_at.desc",
                "limit": str(window * 2),
            },
        )
        return [SupabaseMessageRecord.model_validate(row) for row in reversed(rows)]

    async def insert_user_message(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        content: str,
        raw_query: Optional[str] = None,
        rewritten_query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SupabaseMessageRecord:
        """Insert a user-role turn into an owned conversation."""
        self._verify_owner_context(owner_id)
        if not content or not content.strip():
            raise SupabaseValidationError("Message content cannot be empty.", status_code=400)

        payload: Dict[str, Any] = {
            "conversation_id": str(conversation_id),
            "owner_id": str(owner_id),
            "role": "user",
            "content": content,
            "raw_query": raw_query,
            "rewritten_query": rewritten_query,
            "metadata": metadata if metadata is not None else {},
            "citations": [],
        }
        rows = await self._client.insert("messages", payload=payload)
        if not rows:
            raise SupabaseValidationError("Message insertion returned no record.", status_code=500)
        return SupabaseMessageRecord.model_validate(rows[0])

    async def insert_assistant_message(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> SupabaseMessageRecord:
        """Persist a verified assistant turn strictly via the owner-scoped record_assistant_message RPC."""
        self._verify_owner_context(owner_id)
        if not content or not content.strip():
            raise SupabaseValidationError("Assistant message content cannot be empty.", status_code=400)

        resolved_metadata = metadata if metadata is not None else {}
        resolved_citations = citations if citations is not None else []
        if not isinstance(resolved_metadata, dict):
            raise SupabaseValidationError("Message metadata must be a JSON object.", status_code=400)
        if not isinstance(resolved_citations, list):
            raise SupabaseValidationError("Message citations must be a JSON array.", status_code=400)

        secret = resolve_backend_rpc_hmac_secret(self._rpc_hmac_secret_override)
        payload_digest = compute_record_assistant_message_payload_digest(
            content=content,
            metadata=resolved_metadata,
            citations=resolved_citations,
        )
        proof, nonce, expires_at = build_backend_rpc_proof(
            secret=secret,
            owner_id=owner_id,
            operation="record_assistant_message",
            target_id=conversation_id,
            payload_digest=payload_digest,
        )

        res = await self._client.rpc(
            "record_assistant_message",
            payload={
                "p_conversation_id": str(conversation_id),
                "p_content": content,
                "p_metadata": resolved_metadata,
                "p_citations": resolved_citations,
                "p_backend_proof": proof,
                "p_proof_nonce": nonce,
                "p_proof_expires_at": expires_at,
            },
        )
        row = res[0] if isinstance(res, list) and res else res
        if not isinstance(row, dict):
            raise SupabaseValidationError(
                "Assistant message persistence RPC returned no record.",
                status_code=500,
            )
        return SupabaseMessageRecord.model_validate(row)


def get_message_repository(
    client: SupabaseClient = Depends(get_supabase_client),
) -> SupabaseMessageRepository:
    """FastAPI dependency providing a request-scoped SupabaseMessageRepository."""
    return SupabaseMessageRepository(client=client)
