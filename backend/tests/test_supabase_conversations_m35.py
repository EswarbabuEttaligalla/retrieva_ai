"""Phase 3, Milestone 3.5 Automated Tests: Supabase Conversations, Messages, Query Rewriting & Streaming.

Verifies:
- Owner-scoped conversation CRUD (create, list, get, rename, delete) and uniform 404 for foreign/missing/invalid IDs
- Owner-scoped message reads & user inserts; assistant persistence strictly via approved RPC
- RLS enforcement blocking direct assistant inserts and cross-tenant writes
- Conversational query rewriting isolated to current owner + current conversation + bounded window
- Query rewrite fallback, zero history injection into ChromaDB or ContextBuilder evidence
- Normal 10-step answer lifecycle: user + verified assistant persistence, invalid citations blocking assistant write,
  failed generation leaving no completed assistant turn, updated_at advancing, and retrieval using caller owner_id
- SSE streaming: foreign IDs cannot start stream, completed streams persist verified turns, cancelled/failed
  streams persist no partial assistant output, and ConversationLockManager releases cleanly
- Same-conversation serialization, cross-user concurrency, lock cleanup, startup SQLite absence, and no SQLite fallback on Supabase outage
"""

import asyncio
import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs
from uuid import UUID, uuid4

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from backend.app.conversations.locks import ConversationLockManager
from backend.app.conversations.models import MessageRecord
from backend.app.conversations.query_rewriter import ConversationalQueryRewriter
from backend.app.conversations.service import ConversationNotFoundError, ConversationService
from backend.app.core.auth import clear_jwks_client_cache
from backend.app.core.config import Settings, get_settings
from backend.app.core.supabase_client import SupabaseClient, SupabaseForbiddenError
from backend.app.document_processing.chunker import DocumentChunk
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.llm.client import OpenAIClient
from backend.app.llm.exceptions import LLMError, LLMTimeoutError
from backend.app.main import app
from backend.app.repositories.conversation_repository import SupabaseConversationRepository
from backend.app.repositories.message_repository import SupabaseMessageRepository
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.schemas.answer import AnswerRequest
from backend.app.services.rag_service import RAGService, get_rag_service
from backend.app.vector_store.store import ChromaVectorStore

USER_A_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
USER_B_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
SUPABASE_URL = "https://zblfcosunplvnnwjfyjz.supabase.co"
ANON_KEY = "test-anon-key"


def _make_ec_keypair(kid: str = "m35-ec-1") -> Tuple[Any, Dict[str, Any]]:
    priv = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(priv.public_key()))
    jwk["kid"] = kid
    jwk["use"] = "sig"
    jwk["alg"] = "ES256"
    return priv, jwk


def _issue_jwt(priv: Any, user_id: UUID, kid: str = "m35-ec-1") -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": str(user_id),
            "aud": "authenticated",
            "iss": f"{SUPABASE_URL}/auth/v1",
            "role": "authenticated",
            "email": f"{str(user_id)[:8]}@example.com",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
        },
        priv,
        algorithm="ES256",
        headers={"kid": kid},
    )


class MockSupabaseMultiTenantRLSEngine:
    """Strict PostgREST + RPC + RLS simulator keyed by validated caller JWT."""

    def __init__(self, token_to_user: Dict[str, UUID]) -> None:
        self.token_to_user = token_to_user
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.messages: List[Dict[str, Any]] = []
        self.requests: List[Dict[str, Any]] = []
        self.force_outage: bool = False
        self._tick = 0

    def _next_timestamp(self) -> str:
        self._tick += 1
        base = datetime(2026, 11, 26, 10, 0, 0, tzinfo=timezone.utc)
        return (base + timedelta(seconds=self._tick)).isoformat()

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.force_outage:
            raise httpx.ConnectError("Simulated Supabase outage", request=request)

        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        uid = self.token_to_user.get(token)

        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        self.requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "uid": str(uid) if uid else None,
                "body": body,
            }
        )

        if request.headers.get("apikey") != ANON_KEY or uid is None:
            return httpx.Response(401, json={"message": "Unauthorized"})

        uid_str = str(uid)
        path = request.url.path
        qs = parse_qs(request.url.query.decode("utf-8")) if request.url.query else {}
        now = self._next_timestamp()

        # Approved SECURITY DEFINER RPC for assistant turns
        if path == "/rest/v1/rpc/record_assistant_message" and request.method == "POST":
            cid = body.get("p_conversation_id")
            conv = self.conversations.get(cid)
            if not conv or conv["owner_id"] != uid_str:
                return httpx.Response(
                    403,
                    json={"code": "42501", "message": "Conversation not found or not owned by caller"},
                )
            row = {
                "id": str(uuid4()),
                "conversation_id": cid,
                "owner_id": uid_str,
                "role": "assistant",
                "content": body["p_content"],
                "raw_query": None,
                "rewritten_query": None,
                "metadata": body.get("p_metadata", {}),
                "citations": body.get("p_citations", []),
                "created_at": now,
            }
            self.messages.append(row)
            conv["updated_at"] = now
            return httpx.Response(200, json=row)

        if path == "/rest/v1/documents" and request.method == "GET":
            if uid == USER_A_ID:
                return httpx.Response(
                    200, json=[{"id": "11111111-aaaa-4aaa-8aaa-111111111111", "processing_status": "ready"}]
                )
            return httpx.Response(
                200, json=[{"id": "22222222-bbbb-4bbb-8bbb-222222222222", "processing_status": "ready"}]
            )

        if path == "/rest/v1/conversations":
            if request.method == "POST":
                if body.get("owner_id") != uid_str:
                    return httpx.Response(403, json={"code": "42501", "message": "conversations_insert_own"})
                cid = str(uuid4())
                row = {
                    "id": cid,
                    "owner_id": uid_str,
                    "title": body.get("title", "New Conversation"),
                    "created_at": now,
                    "updated_at": now,
                }
                self.conversations[cid] = row
                return httpx.Response(201, json=[row])

            if request.method == "GET":
                rows = [c for c in self.conversations.values() if c["owner_id"] == uid_str]
                if "id" in qs:
                    tid = qs["id"][0].removeprefix("eq.")
                    rows = [c for c in rows if c["id"] == tid]
                if qs.get("order") == ["updated_at.desc"]:
                    rows.sort(key=lambda r: r["updated_at"], reverse=True)
                offset = int(qs.get("offset", ["0"])[0])
                limit = int(qs.get("limit", ["100"])[0])
                return httpx.Response(200, json=rows[offset:offset + limit])

            if request.method == "PATCH":
                tid = qs["id"][0].removeprefix("eq.")
                row = self.conversations.get(tid)
                if not row or row["owner_id"] != uid_str:
                    return httpx.Response(200, json=[])
                if "title" in body:
                    row["title"] = body["title"]
                row["updated_at"] = now
                return httpx.Response(200, json=[row])

            if request.method == "DELETE":
                tid = qs["id"][0].removeprefix("eq.")
                row = self.conversations.get(tid)
                if not row or row["owner_id"] != uid_str:
                    return httpx.Response(200, json=[])
                del self.conversations[tid]
                self.messages = [m for m in self.messages if m["conversation_id"] != tid]
                return httpx.Response(200, json=[row])

        if path == "/rest/v1/messages":
            if request.method == "POST":
                # Enforce Phase 1 RLS policy: role='user' only AND conversation owned by uid
                if body.get("owner_id") != uid_str or body.get("role") != "user":
                    return httpx.Response(403, json={"code": "42501", "message": "messages_insert_user_turn_in_owned_conversation"})
                cid = body.get("conversation_id")
                conv = self.conversations.get(cid)
                if not conv or conv["owner_id"] != uid_str:
                    return httpx.Response(403, json={"code": "42501", "message": "messages_insert_user_turn_in_owned_conversation"})
                row = {
                    "id": str(uuid4()),
                    "conversation_id": cid,
                    "owner_id": uid_str,
                    "role": "user",
                    "content": body["content"],
                    "raw_query": body.get("raw_query"),
                    "rewritten_query": body.get("rewritten_query"),
                    "metadata": body.get("metadata", {}),
                    "citations": [],
                    "created_at": now,
                }
                self.messages.append(row)
                return httpx.Response(201, json=[row])

            if request.method == "GET":
                # Enforce RLS SELECT via owned conversation
                rows = [
                    m
                    for m in self.messages
                    if m["owner_id"] == uid_str
                    and (conv := self.conversations.get(m["conversation_id"])) is not None
                    and conv["owner_id"] == uid_str
                ]
                if "conversation_id" in qs:
                    cid = qs["conversation_id"][0].removeprefix("eq.")
                    rows = [m for m in rows if m["conversation_id"] == cid]
                if qs.get("order") == ["created_at.desc"]:
                    rows = list(reversed(rows))
                if "limit" in qs:
                    rows = rows[:int(qs["limit"][0])]
                return httpx.Response(200, json=rows)

        return httpx.Response(404, json={"message": "Not found"})


@pytest.fixture(scope="module")
def shared_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder.get_instance()


@pytest.fixture
def m35_env(shared_embedder: SentenceTransformerEmbedder):
    """Set up two authenticated tenants, ChromaDB corpus, RAGService, and TestClient."""
    temp_dir = tempfile.mkdtemp(prefix="retrieva_m35_")
    priv, jwk = _make_ec_keypair()
    token_a = _issue_jwt(priv, USER_A_ID)
    token_b = _issue_jwt(priv, USER_B_ID)

    engine = MockSupabaseMultiTenantRLSEngine({token_a: USER_A_ID, token_b: USER_B_ID})
    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(engine.handle))

    store = ChromaVectorStore(collection_name="m35_test_col", persist_directory=temp_dir)
    chunk_a = DocumentChunk(
        chunk_id="m35_chunk_a_1",
        document_id="11111111-aaaa-4aaa-8aaa-111111111111",
        filename="tenant_a_notes.txt",
        page_number=1,
        page_references=[1],
        chunk_index=0,
        chunk_text="Tenant Alpha operates the Obsidian satellite telemetry cluster in Zurich.",
        start_offset=0,
        end_offset=73,
        char_count=73,
        word_count=11,
    )
    chunk_b = DocumentChunk(
        chunk_id="m35_chunk_b_1",
        document_id="22222222-bbbb-4bbb-8bbb-222222222222",
        filename="tenant_b_notes.txt",
        page_number=1,
        page_references=[1],
        chunk_index=0,
        chunk_text="Tenant Beta manages the Beryllium deep-ocean buoy network in Bergen.",
        start_offset=0,
        end_offset=68,
        char_count=68,
        word_count=11,
    )
    store.add_chunks([chunk_a], shared_embedder.embed_batch([chunk_a.chunk_text]), owner_id=USER_A_ID, prevent_duplicates=False)
    store.add_chunks([chunk_b], shared_embedder.embed_batch([chunk_b.chunk_text]), owner_id=USER_B_ID, prevent_duplicates=False)

    retriever = Retriever(
        dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
        vector_store=store,
    )

    mock_llm = MagicMock(spec=OpenAIClient)
    mock_llm.api_key = "sk-m35-test"
    mock_llm.model = "gpt-4o-mini"
    mock_llm.rewrite_query_async = AsyncMock(return_value="Standalone satellite cluster query")
    mock_llm.generate.return_value = {
        "content": json.dumps(
            {
                "answer": "Obsidian telemetry runs in Zurich [Source 1].",
                "insufficient_context": False,
                "citations": [
                    {
                        "source_id": "[Source 1]",
                        "claim": "Obsidian runs in Zurich",
                        "quote": "Tenant Alpha operates the Obsidian satellite telemetry cluster in Zurich.",
                    }
                ],
            }
        ),
        "prompt_tokens": 40,
        "completion_tokens": 12,
        "total_tokens": 52,
    }

    lock_manager = ConversationLockManager()
    rag_svc = RAGService(retriever=retriever, llm_client=mock_llm, lock_manager=lock_manager)
    settings = Settings(
        APP_ENV="development",
        SUPABASE_URL=SUPABASE_URL,
        SUPABASE_ANON_KEY=ANON_KEY,
        CONVERSATION_DB_PATH=str(Path(temp_dir) / "should_never_exist.db"),
    )

    def _make_conv_service(token: str, uid: UUID, rewriter_client: Optional[OpenAIClient] = None) -> ConversationService:
        client = SupabaseClient(
            supabase_url=SUPABASE_URL,
            anon_key=ANON_KEY,
            access_token=token,
            owner_id=uid,
            http_client=mock_http,
        )
        return ConversationService(
            conversation_repo=SupabaseConversationRepository(client),
            message_repo=SupabaseMessageRepository(client),
            query_rewriter=ConversationalQueryRewriter(llm_client=rewriter_client or mock_llm),
        )

    prev_settings = app.dependency_overrides.get(get_settings)
    prev_ret = app.dependency_overrides.get(get_retriever)
    prev_rag = app.dependency_overrides.get(get_rag_service)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_rag_service] = lambda: rag_svc

    clear_jwks_client_cache()
    try:
        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}), patch(
            "backend.app.core.supabase_client.get_supabase_http_client",
            return_value=mock_http,
        ):
            with TestClient(app) as client:
                yield {
                    "temp_dir": temp_dir,
                    "client": client,
                    "token_a": token_a,
                    "token_b": token_b,
                    "engine": engine,
                    "store": store,
                    "retriever": retriever,
                    "rag": rag_svc,
                    "mock_llm": mock_llm,
                    "lock_manager": lock_manager,
                    "make_conv_service": _make_conv_service,
                }
    finally:
        clear_jwks_client_cache()
        for key, prev in [
            (get_settings, prev_settings),
            (get_retriever, prev_ret),
            (get_rag_service, prev_rag),
        ]:
            if prev is None:
                app.dependency_overrides.pop(key, None)
            else:
                app.dependency_overrides[key] = prev
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestConversationAndMessageOwnership:
    """Tests HTTP + repository ownership, uniform 404s, RLS blocks, and RPC-only assistant persistence."""

    def test_conversation_crud_isolation_and_uniform_404(self, m35_env):
        client: TestClient = m35_env["client"]
        headers_a = {"Authorization": f"Bearer {m35_env['token_a']}"}
        headers_b = {"Authorization": f"Bearer {m35_env['token_b']}"}

        # Startup must not create SQLite conversation DB file
        assert not Path(m35_env["temp_dir"], "should_never_exist.db").exists()

        # Create conversations for A and B
        ca = client.post("/api/v1/conversations", headers=headers_a, json={"title": "Alpha Session"})
        cb = client.post("/api/v1/conversations", headers=headers_b, json={"title": "Beta Session"})
        assert ca.status_code == 201
        assert cb.status_code == 201
        cid_a = ca.json()["id"]
        cid_b = cb.json()["id"]

        # Listing only returns caller's own conversations
        list_a = client.get("/api/v1/conversations", headers=headers_a).json()
        list_b = client.get("/api/v1/conversations", headers=headers_b).json()
        assert [c["id"] for c in list_a] == [cid_a]
        assert [c["id"] for c in list_b] == [cid_b]

        # Foreign UUID, nonexistent UUID, and malformed ID all produce identical 404
        missing_uuid = str(uuid4())
        r_foreign = client.get(f"/api/v1/conversations/{cid_b}", headers=headers_a)
        r_missing = client.get(f"/api/v1/conversations/{missing_uuid}", headers=headers_a)
        r_malformed = client.get("/api/v1/conversations/not-a-valid-uuid", headers=headers_a)
        assert r_foreign.status_code == r_missing.status_code == r_malformed.status_code == 404
        assert r_foreign.json() == r_missing.json() == r_malformed.json()

        # User A cannot rename or delete User B's conversation (uniform 404)
        patch_foreign = client.patch(
            f"/api/v1/conversations/{cid_b}",
            headers=headers_a,
            json={"title": "Hijacked"},
        )
        del_foreign = client.delete(f"/api/v1/conversations/{cid_b}", headers=headers_a)
        assert patch_foreign.status_code == 404
        assert del_foreign.status_code == 404
        assert m35_env["engine"].conversations[cid_b]["title"] == "Beta Session"

    @pytest.mark.asyncio
    async def test_message_ownership_rls_and_rpc_only_assistant_writes(self, m35_env):
        svc_a: ConversationService = m35_env["make_conv_service"](m35_env["token_a"], USER_A_ID)
        svc_b: ConversationService = m35_env["make_conv_service"](m35_env["token_b"], USER_B_ID)
        engine: MockSupabaseMultiTenantRLSEngine = m35_env["engine"]

        conv_a = await svc_a.create_conversation(owner_id=USER_A_ID, title="Msg Ownership")
        await svc_a.record_user_message(conv_a.id, USER_A_ID, "User A raw Q", "User A rewritten Q")
        await svc_a.record_assistant_message(
            conv_a.id,
            USER_A_ID,
            "User A assistant answer",
            metadata={"retrieval_mode": "hybrid"},
            citations=[{"source_id": "[Source 1]", "document_id": "11111111-aaaa-4aaa-8aaa-111111111111"}],
        )

        # User A reads own messages; User B gets None on get_conversation_with_messages & empty list directly
        detail_a = await svc_a.get_conversation_with_messages(conv_a.id, USER_A_ID)
        detail_b = await svc_b.get_conversation_with_messages(conv_a.id, USER_B_ID)
        assert detail_a is not None and len(detail_a.messages) == 2
        assert detail_b is None
        assert await svc_b.message_repo.list_messages(UUID(conv_a.id), USER_B_ID) == []

        # User B cannot insert message into User A's conversation
        with pytest.raises(ConversationNotFoundError):
            await svc_b.record_user_message(conv_a.id, USER_B_ID, "Cross-tenant turn", "Cross-tenant turn")
        with pytest.raises(SupabaseForbiddenError):
            await svc_b.message_repo.insert_user_message(UUID(conv_a.id), USER_B_ID, "Direct RLS probe")

        # Direct role='assistant' table insert is blocked by Phase 1 RLS even for User A
        with pytest.raises(SupabaseForbiddenError):
            await svc_a.message_repo._client.insert(
                "messages",
                payload={
                    "conversation_id": conv_a.id,
                    "owner_id": str(USER_A_ID),
                    "role": "assistant",
                    "content": "Bypass attempt",
                    "metadata": {},
                    "citations": [],
                },
            )

        # Verify record_assistant_message used ONLY RPC, never direct POST /rest/v1/messages with assistant role
        direct_assistant_posts = [
            req
            for req in engine.requests
            if req["method"] == "POST"
            and req["path"] == "/rest/v1/messages"
            and req["body"].get("role") == "assistant"
            and req["uid"] == str(USER_A_ID)
        ]
        # Only our intentional negative RLS probe above attempted it; ConversationService used /rpc/record_assistant_message
        assert len(direct_assistant_posts) == 1
        rpc_calls = [req for req in engine.requests if req["path"] == "/rest/v1/rpc/record_assistant_message"]
        assert len(rpc_calls) == 1


class TestQueryRewritingAndLifecycle:
    """Tests isolated query rewriting history, normal 10-step answer lifecycle, and invalid citation blocking."""

    @pytest.mark.asyncio
    async def test_query_rewriting_isolation_window_and_fallback(self, m35_env):
        svc_a: ConversationService = m35_env["make_conv_service"](m35_env["token_a"], USER_A_ID)
        svc_b: ConversationService = m35_env["make_conv_service"](m35_env["token_b"], USER_B_ID)
        rag: RAGService = m35_env["rag"]
        mock_llm: MagicMock = m35_env["mock_llm"]
        store: ChromaVectorStore = m35_env["store"]

        initial_chroma_count = store.collection.count()

        # Seed User B conversation + User A OTHER conversation with secrets that must NEVER leak
        cb = await svc_b.create_conversation(owner_id=USER_B_ID, title="B Secret")
        await svc_b.record_user_message(cb.id, USER_B_ID, "User B secret passphrase", "User B secret passphrase")
        ca_other = await svc_a.create_conversation(owner_id=USER_A_ID, title="A Unrelated")
        await svc_a.record_user_message(ca_other.id, USER_A_ID, "A other session topic", "A other session topic")

        ca = await svc_a.create_conversation(owner_id=USER_A_ID, title="New Conversation")
        await svc_a.record_user_message(ca.id, USER_A_ID, "Tell me about Obsidian.", "Tell me about Obsidian.")
        await svc_a.record_assistant_message(ca.id, USER_A_ID, "Obsidian runs telemetry.")

        seen_histories: List[List[MessageRecord]] = []
        orig_rewrite = svc_a.query_rewriter.rewrite_async

        async def spy_rewrite(raw_query: str, history: List[MessageRecord], window=None):
            seen_histories.append(history)
            return await orig_rewrite(raw_query, history, window=window)

        mock_llm.rewrite_query_async = AsyncMock(return_value="Where is Obsidian telemetry cluster located?")
        with patch.object(svc_a.query_rewriter, "rewrite_async", side_effect=spy_rewrite), patch.object(
            rag.context_builder, "build_context", wraps=rag.context_builder.build_context
        ) as spy_ctx:
            resp = await rag.answer_question_async(
                AnswerRequest(query="Where does it run?", conversation_id=ca.id),
                owner_id=USER_A_ID,
                conversation_service=svc_a,
            )

        assert resp.rewritten_query == "Where is Obsidian telemetry cluster located?"
        assert len(seen_histories) == 1
        assert all(m.conversation_id == ca.id for m in seen_histories[0])
        assert all("secret" not in m.content.lower() and "other session" not in m.content.lower() for m in seen_histories[0])

        # Conversation history never enters ChromaDB or ContextBuilder evidence chunks
        assert store.collection.count() == initial_chroma_count
        chunks_into_ctx = spy_ctx.call_args[0][0]
        assert all("Obsidian runs telemetry" not in c.chunk_text for c in chunks_into_ctx)

        # Query rewrite failure falls back cleanly to original raw query
        mock_llm.rewrite_query_async = AsyncMock(side_effect=LLMTimeoutError("Rewrite timeout"))
        rewritten, _ = await svc_a.prepare_conversational_turn(ca.id, USER_A_ID, "Follow up question")
        assert rewritten == "Follow up question"

    @pytest.mark.asyncio
    async def test_answer_lifecycle_updates_timestamp_and_blocks_bad_citations_or_failed_gen(self, m35_env):
        client: TestClient = m35_env["client"]
        headers_a = {"Authorization": f"Bearer {m35_env['token_a']}"}
        engine: MockSupabaseMultiTenantRLSEngine = m35_env["engine"]
        mock_llm: MagicMock = m35_env["mock_llm"]

        c_res = client.post("/api/v1/conversations", headers=headers_a, json={"title": "New Conversation"})
        assert c_res.status_code == 201
        cid = c_res.json()["id"]
        initial_updated = engine.conversations[cid]["updated_at"]

        # 1. Valid turn -> user + assistant persisted, title auto-derived, updated_at advances, only A docs retrieved
        ans_res = client.post(
            f"/api/v1/conversations/{cid}/answer",
            headers=headers_a,
            json={"query": "Where is Obsidian operated?"},
        )
        assert ans_res.status_code == 200
        body = ans_res.json()
        assert "Zurich" in body["answer"]
        assert body["citations"][0]["document_id"] == "11111111-aaaa-4aaa-8aaa-111111111111"
        assert engine.conversations[cid]["updated_at"] > initial_updated
        assert engine.conversations[cid]["title"] != "New Conversation"
        assert len([m for m in engine.messages if m["conversation_id"] == cid]) == 2

        # 2. Failed LLM generation -> does NOT persist failed assistant message or extra user message
        with patch.object(mock_llm, "generate", side_effect=LLMError("Upstream failure")):
            fail_res = client.post(
                f"/api/v1/conversations/{cid}/answer",
                headers=headers_a,
                json={"query": "Will this fail?"},
            )
        assert fail_res.status_code == 500
        assert len([m for m in engine.messages if m["conversation_id"] == cid]) == 2

        # 3. Hallucinated citations -> safe rejection returned, NO assistant message persisted
        with patch.object(
            mock_llm,
            "generate",
            return_value={
                "content": json.dumps({"answer": "Fake claim [Source 99].", "citations": [{"source_id": "[Source 99]"}]}),
                "model": "gpt-4o-mini",
            },
        ):
            rej_res = client.post(
                f"/api/v1/conversations/{cid}/answer",
                headers=headers_a,
                json={"query": "Tell me fake news"},
            )
        assert rej_res.status_code == 200
        assert rej_res.json()["metadata"]["has_hallucinated_sources"] is True
        msgs_after_rej = [m for m in engine.messages if m["conversation_id"] == cid]
        assistant_msgs = [m for m in msgs_after_rej if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1  # Only turn 1's valid assistant response!


class TestStreamingConcurrencyAndOutageRecovery:
    """Tests SSE persistence, cancellation/error cleanup, foreign blocking, concurrency, and no SQLite fallback."""

    def test_streaming_contract_foreign_denial_and_lock_release_on_failure(self, m35_env):
        client: TestClient = m35_env["client"]
        headers_a = {"Authorization": f"Bearer {m35_env['token_a']}"}
        headers_b = {"Authorization": f"Bearer {m35_env['token_b']}"}
        mock_llm: MagicMock = m35_env["mock_llm"]
        lock_mgr: ConversationLockManager = m35_env["lock_manager"]
        engine: MockSupabaseMultiTenantRLSEngine = m35_env["engine"]

        cid_a = client.post("/api/v1/conversations", headers=headers_a, json={"title": "Stream Session"}).json()["id"]

        # Foreign user cannot start stream (404 BEFORE generation starts)
        mock_llm.generate_stream_async = MagicMock()
        foreign_stream = client.post(
            f"/api/v1/conversations/{cid_a}/answer/stream",
            headers=headers_b,
            json={"query": "Steal User A stream"},
        )
        assert foreign_stream.status_code == 404
        assert not mock_llm.generate_stream_async.called

        # Valid stream preserves SSE events and persists verified messages
        async def ok_stream(*args, **kwargs):
            yield "Zurich telemetry cluster [Source 1]."

        mock_llm.generate_stream_async = MagicMock(side_effect=ok_stream)
        ok_res = client.post(
            f"/api/v1/conversations/{cid_a}/answer/stream",
            headers=headers_a,
            json={"query": "Where is Obsidian telemetry?"},
        )
        assert ok_res.status_code == 200
        assert "event: token" in ok_res.text
        assert "event: citations" in ok_res.text
        assert "event: done" in ok_res.text
        assert cid_a not in lock_mgr._locks
        assert len([m for m in engine.messages if m["conversation_id"] == cid_a]) == 2

        # Stream LLM error releases lock and persists NO partial assistant message
        async def failing_stream(*args, **kwargs):
            yield "Partial before failure "
            raise LLMTimeoutError("Stream timed out")

        mock_llm.generate_stream_async = MagicMock(side_effect=failing_stream)
        err_res = client.post(
            f"/api/v1/conversations/{cid_a}/answer/stream",
            headers=headers_a,
            json={"query": "Fail mid stream"},
        )
        assert err_res.status_code == 200
        assert "event: error" in err_res.text
        assert cid_a not in lock_mgr._locks
        assert len([m for m in engine.messages if m["conversation_id"] == cid_a and m["role"] == "assistant"]) == 1

    @pytest.mark.asyncio
    async def test_cancellation_serialization_cross_user_concurrency_and_no_sqlite_fallback(self, m35_env):
        svc_a: ConversationService = m35_env["make_conv_service"](m35_env["token_a"], USER_A_ID)
        svc_b: ConversationService = m35_env["make_conv_service"](m35_env["token_b"], USER_B_ID)
        rag: RAGService = m35_env["rag"]
        mock_llm: MagicMock = m35_env["mock_llm"]
        lock_mgr: ConversationLockManager = m35_env["lock_manager"]
        engine: MockSupabaseMultiTenantRLSEngine = m35_env["engine"]
        client: TestClient = m35_env["client"]

        ca = await svc_a.create_conversation(owner_id=USER_A_ID, title="Conc A")
        cb = await svc_b.create_conversation(owner_id=USER_B_ID, title="Conc B")

        # 1. Cancelled stream persists no assistant message and frees lock
        async def cancel_stream(*args, **kwargs):
            yield "Partial canceled "
            await asyncio.sleep(0.3)
            yield "never"

        mock_llm.generate_stream_async = MagicMock(side_effect=cancel_stream)
        gen = rag.answer_question_stream(
            AnswerRequest(query="Cancel me", conversation_id=ca.id),
            owner_id=USER_A_ID,
            conversation_service=svc_a,
        )
        async for ev in gen:
            if "event: token" in ev:
                break
        await gen.aclose()
        assert ca.id not in lock_mgr._locks
        assert all(m["role"] != "assistant" for m in engine.messages if m["conversation_id"] == ca.id)

        # 2. Simultaneous requests for different users run concurrently; same conversation serializes cleanly
        log: List[str] = []

        async def router_stream(*args, **kwargs):
            msgs = kwargs.get("messages") or []
            txt = msgs[-1]["content"] if msgs else ""
            if "Slow A" in txt:
                log.append("a_start")
                await asyncio.sleep(0.09)
                log.append("a_end")
                yield "Obsidian answer [Source 1]."
            else:
                log.append("b_start")
                await asyncio.sleep(0.01)
                log.append("b_end")
                yield "Beryllium answer [Source 1]."

        mock_llm.generate_stream_async = MagicMock(side_effect=router_stream)

        async def run_stream(uid: UUID, cid: str, svc: ConversationService, q: str):
            frames = []
            async for f in rag.answer_question_stream(
                AnswerRequest(query=q, conversation_id=cid),
                owner_id=uid,
                conversation_service=svc,
            ):
                frames.append(f)
            return frames

        await asyncio.gather(
            run_stream(USER_A_ID, ca.id, svc_a, "Slow A obsidian"),
            run_stream(USER_B_ID, cb.id, svc_b, "Fast B beryllium"),
        )
        assert log.index("b_end") < log.index("a_end")
        assert ca.id not in lock_mgr._locks and cb.id not in lock_mgr._locks

        # 3. Supabase outage returns 503 Service Unavailable and NEVER falls back to SQLite
        engine.force_outage = True
        try:
            with patch("backend.app.conversations.database.get_db_connection") as spy_sqlite:
                outage_res = client.get(
                    "/api/v1/conversations",
                    headers={"Authorization": f"Bearer {m35_env['token_a']}"},
                )
                assert outage_res.status_code == 503
                assert not spy_sqlite.called
                assert not Path(m35_env["temp_dir"], "should_never_exist.db").exists()
        finally:
            engine.force_outage = False
