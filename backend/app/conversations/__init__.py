"""Conversations module for Supabase-backed chat sessions, message history, and query rewriting."""

from backend.app.conversations.models import (
    ConversationDetail,
    ConversationRecord,
    CreateConversationRequest,
    MessageRecord,
)
from backend.app.conversations.query_rewriter import ConversationalQueryRewriter, get_query_rewriter
from backend.app.conversations.service import (
    ConversationNotFoundError,
    ConversationService,
    get_conversation_service,
)

__all__ = [
    "ConversationDetail",
    "ConversationNotFoundError",
    "ConversationRecord",
    "ConversationService",
    "ConversationalQueryRewriter",
    "CreateConversationRequest",
    "MessageRecord",
    "get_conversation_service",
    "get_query_rewriter",
]
