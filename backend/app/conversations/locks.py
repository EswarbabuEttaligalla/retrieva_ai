"""Per-conversation request serialization and concurrency management."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional

logger = logging.getLogger("retrieva_ai.conversations.locks")


class ConversationLockManager:
    """Manages dynamic per-conversation asyncio locks to serialize turns targeting the same session.

    Different conversations execute concurrently without blocking each other.
    When all active requests for a conversation complete, its lock is safely pruned from memory.
    """

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._waiters: Dict[str, int] = {}
        self._mutex = asyncio.Lock()

    async def acquire(self, conversation_id: Optional[str]) -> Optional[str]:
        """Acquire the lock for a given conversation_id. Returns the cleaned id if acquired."""
        if not conversation_id:
            return None
        cid = str(conversation_id).strip()
        if not cid:
            return None

        async with self._mutex:
            self._waiters[cid] = self._waiters.get(cid, 0) + 1
            if cid not in self._locks:
                self._locks[cid] = asyncio.Lock()
            lock = self._locks[cid]

        await lock.acquire()
        return cid

    async def release(self, conversation_id: Optional[str]) -> None:
        """Release the lock for a given conversation_id and clean up if no waiters remain."""
        if not conversation_id:
            return
        cid = str(conversation_id).strip()
        if not cid:
            return

        async with self._mutex:
            if cid in self._locks:
                try:
                    self._locks[cid].release()
                except RuntimeError as exc:
                    logger.debug("Notice on lock release: %s", exc)
            self._waiters[cid] = max(0, self._waiters.get(cid, 1) - 1)
            if self._waiters[cid] == 0:
                self._locks.pop(cid, None)
                self._waiters.pop(cid, None)

    @asynccontextmanager
    async def lock(self, conversation_id: Optional[str]) -> AsyncGenerator[None, None]:
        """Context manager acquiring and releasing the exclusive lock for a given conversation_id."""
        cid = await self.acquire(conversation_id)
        try:
            yield
        finally:
            await self.release(cid)


_lock_manager_instance: Optional[ConversationLockManager] = None


def get_conversation_lock_manager() -> ConversationLockManager:
    """Return singleton instance of ConversationLockManager."""
    global _lock_manager_instance
    if _lock_manager_instance is None:
        _lock_manager_instance = ConversationLockManager()
    return _lock_manager_instance
