"""Conversations module for persistent chat sessions, message history, and conversational query rewriting."""

from backend.app.conversations.models import (
    ConversationRecord,
    ConversationDetail,
    MessageRecord,
    CreateConversationRequest,
)
from backend.app.conversations.repository import ConversationRepository, get_conversation_repository
from backend.app.conversations.service import ConversationService, get_conversation_service
from backend.app.conversations.query_rewriter import ConversationalQueryRewriter, get_query_rewriter

__all__ = [
    "ConversationRecord",
    "ConversationDetail",
    "MessageRecord",
    "CreateConversationRequest",
    "ConversationRepository",
    "get_conversation_repository",
    "ConversationService",
    "get_conversation_service",
    "ConversationalQueryRewriter",
    "get_query_rewriter",
]
