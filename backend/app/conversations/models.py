"""Pydantic schemas and domain models for persistent conversations and chat memory."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from backend.app.schemas.retrieval import RetrievalConfiguration


class MessageRole(str, Enum):
    """Permitted message participant roles."""
    USER = "user"
    ASSISTANT = "assistant"


class MessageRecord(BaseModel):
    """Domain model representing a single conversation message."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique message identifier (UUID)")
    conversation_id: str = Field(description="Foreign key pointing to parent conversation")
    role: str = Field(description="Role: 'user' or 'assistant'")
    content: str = Field(description="Verbatim message content or answer")
    raw_query: Optional[str] = Field(default=None, description="Original user query if role=user")
    rewritten_query: Optional[str] = Field(default=None, description="Standalone rewritten retrieval query if role=user")
    created_at: str = Field(description="UTC ISO-8601 creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Serialized execution metadata, citations, and diagnostics")


class ConversationRecord(BaseModel):
    """Summary record for a conversation listed in the sidebar or catalog."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique conversation identifier (UUID)")
    title: str = Field(description="Human-readable conversation title")
    created_at: str = Field(description="UTC ISO-8601 creation timestamp")
    updated_at: str = Field(description="UTC ISO-8601 last update timestamp")
    message_count: int = Field(default=0, description="Total number of persisted messages in conversation")


class ConversationDetail(BaseModel):
    """Detailed conversation record containing full chronological message history."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique conversation identifier (UUID)")
    title: str = Field(description="Human-readable conversation title")
    created_at: str = Field(description="UTC ISO-8601 creation timestamp")
    updated_at: str = Field(description="UTC ISO-8601 last update timestamp")
    messages: List[MessageRecord] = Field(default_factory=list, description="Chronological sequence of conversation turns")


class CreateConversationRequest(BaseModel):
    """Payload schema for creating a new conversation."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(default=None, max_length=200, description="Optional initial conversation title")


class UpdateConversationRequest(BaseModel):
    """Payload schema for updating conversation metadata."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200, description="New conversation title")


class ConversationAnswerRequest(BaseModel):
    """Payload schema for generating an answer within a specific conversation."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000, description="Natural language user question")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document filter")
    retrieval_config: Optional[RetrievalConfiguration] = Field(default=None, description="Optional retrieval configuration")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, ge=50, le=4000, description="Maximum completion tokens")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response via Server-Sent Events (SSE)")
