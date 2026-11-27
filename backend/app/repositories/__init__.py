"""Owner-scoped Supabase repositories for Retrieva AI."""

from backend.app.repositories.conversation_repository import (
    SupabaseConversationRecord,
    SupabaseConversationRepository,
    get_supabase_conversation_repository,
)
from backend.app.repositories.document_repository import (
    DocumentRecord,
    DocumentRepository,
    get_document_repository,
)
from backend.app.repositories.message_repository import (
    SupabaseMessageRecord,
    SupabaseMessageRepository,
    get_message_repository,
)
from backend.app.repositories.storage_repository import (
    PRIVATE_BUCKET_NAME,
    SupabaseStorageRepository,
    get_storage_repository,
)

__all__ = [
    "DocumentRecord",
    "DocumentRepository",
    "PRIVATE_BUCKET_NAME",
    "SupabaseConversationRecord",
    "SupabaseConversationRepository",
    "SupabaseMessageRecord",
    "SupabaseMessageRepository",
    "SupabaseStorageRepository",
    "get_document_repository",
    "get_message_repository",
    "get_storage_repository",
    "get_supabase_conversation_repository",
]
