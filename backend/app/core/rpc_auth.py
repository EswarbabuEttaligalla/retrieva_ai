"""Shared backend-only HMAC-SHA256 RPC proof generation for Supabase SECURITY DEFINER RPCs."""

import hashlib
import hmac
import math
import secrets
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from backend.app.core.config import get_settings
from backend.app.core.supabase_client import SupabaseConfigError

RPC_PROOF_TTL_SECONDS = 60


def lp_field(value: Optional[str]) -> str:
    """Length-prefix a nullable text field to match private.lp_field() in PostgreSQL."""
    text = "" if value is None else str(value)
    return f"{len(text)}:{text}"


def _canonical_decimal_str(value: Union[int, float]) -> str:
    """Format int/float as a plain decimal without scientific notation or trailing '.0'."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite.")
        text = format(Decimal(str(value)), "f")
    else:
        text = str(value)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text


def canonical_jsonb(value: Any) -> str:
    """Serialize JSON identically to private.canonical_jsonb(jsonb) in PostgreSQL."""
    if value is None:
        return "n"
    if isinstance(value, bool):
        return "b:true" if value else "b:false"
    if isinstance(value, (int, float)):
        return f"d:{_canonical_decimal_str(value)}"
    if isinstance(value, str):
        return f"s:{lp_field(value)}"
    if isinstance(value, list):
        return "a:[" + ",".join(canonical_jsonb(item) for item in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]).encode("utf-8"))
        inner = ",".join(f"{lp_field(str(k))}={canonical_jsonb(v)}" for k, v in items)
        return f"o:{{{inner}}}"
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


canonical_pg_jsonb_text = canonical_jsonb


def resolve_backend_rpc_hmac_secret(rpc_hmac_secret: Optional[str]) -> str:
    """Resolve and validate the server-only HMAC secret, failing closed when missing."""
    resolved = get_settings().BACKEND_RPC_HMAC_SECRET if rpc_hmac_secret is None else rpc_hmac_secret
    cleaned = resolved.strip() if isinstance(resolved, str) else ""
    if not cleaned:
        raise SupabaseConfigError(
            "BACKEND_RPC_HMAC_SECRET must be configured to invoke protected Supabase RPCs."
        )
    return cleaned


def compute_create_document_payload_digest(
    *,
    storage_path: str,
    filename: str,
    file_extension: str,
    mime_type: str,
    content_hash: str,
    file_size_bytes: int,
    processing_status: str,
    embedding_model: Optional[str],
) -> str:
    """Compute canonical SHA-256 digest for create_user_document parameters."""
    canonical = "|".join(
        [
            lp_field(storage_path),
            lp_field(filename),
            lp_field(file_extension.lower()),
            lp_field(mime_type.lower()),
            lp_field(content_hash.lower()),
            lp_field(str(file_size_bytes)),
            lp_field(processing_status),
            lp_field(embedding_model),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_update_status_payload_digest(
    *,
    processing_status: str,
    total_pages: Optional[int],
    total_chunks: Optional[int],
    embedding_model: Optional[str],
    error_message: Optional[str],
) -> str:
    """Compute canonical SHA-256 digest for update_user_document_status parameters."""
    canonical = "|".join(
        [
            lp_field(processing_status),
            lp_field(None if total_pages is None else str(total_pages)),
            lp_field(None if total_chunks is None else str(total_chunks)),
            lp_field(embedding_model),
            lp_field(error_message),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_delete_document_payload_digest(document_id: UUID) -> str:
    """Compute canonical SHA-256 digest for delete_user_document parameters."""
    return hashlib.sha256(lp_field(str(document_id)).encode("utf-8")).hexdigest()


def compute_record_assistant_message_payload_digest(
    *,
    content: str,
    metadata: Optional[Dict[str, Any]],
    citations: Optional[List[Dict[str, Any]]],
) -> str:
    """Compute canonical SHA-256 digest for record_assistant_message parameters."""
    resolved_metadata = metadata if metadata is not None else {}
    resolved_citations = citations if citations is not None else []
    canonical = "|".join(
        [
            lp_field(content),
            lp_field(canonical_jsonb(resolved_metadata)),
            lp_field(canonical_jsonb(resolved_citations)),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_backend_rpc_proof(
    *,
    secret: str,
    owner_id: UUID,
    operation: str,
    target_id: Optional[UUID] = None,
    document_id: Optional[UUID] = None,
    payload_digest: str,
    ttl_seconds: int = RPC_PROOF_TTL_SECONDS,
) -> Tuple[str, str, int]:
    """Generate (proof_hex, nonce, expires_at) bound to user, operation, target UUID, and payload."""
    bound_id = target_id if target_id is not None else document_id
    if bound_id is None:
        raise ValueError("target_id or document_id must be provided when building an RPC proof.")
    expires_at = int(time.time()) + ttl_seconds
    nonce = secrets.token_hex(16)
    message = f"{owner_id}|{operation}|{bound_id}|{payload_digest}|{expires_at}|{nonce}"
    proof = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return proof, nonce, expires_at
