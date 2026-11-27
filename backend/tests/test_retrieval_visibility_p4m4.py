"""Phase 4 Milestone 4: Retrieval visibility and consistency hardening tests."""

import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs
from uuid import UUID, uuid4

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from backend.app.core.auth import clear_jwks_client_cache, get_current_user
from backend.app.core.config import Settings, get_settings
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.llm.client import OpenAIClient
from backend.app.main import app
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.retrieval.tenant_bm25 import TenantBM25Manager
from backend.app.schemas.document import DocumentChunk
from backend.app.services.rag_service import RAGService, get_rag_service
from backend.app.vector_store.store import ChromaVectorStore

USER_A_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
USER_B_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
SUPABASE_URL = "https://zblfcosunplvnnwjfyjz.supabase.co"
ANON_KEY = "test-public-anon-key"


class VisibilitySupabaseMock:
    """In-memory PostgREST simulator supporting document status transitions and outage injection."""

    def __init__(self, token_to_user: Dict[str, UUID]) -> None:
        self.token_to_user = token_to_user
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.fail_select: bool = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        tok = auth.removeprefix("Bearer ").strip()
        uid = self.token_to_user.get(tok)
        if uid is None:
            return httpx.Response(401, json={"message": "Unauthorized"})

        if request.url.path == "/rest/v1/documents" and request.method == "GET":
            if self.fail_select:
                return httpx.Response(503, json={"message": "Simulated Supabase outage"})
            qs = parse_qs(request.url.query.decode("utf-8")) if request.url.query else {}
            rows = [r for r in self.documents.values() if r["owner_id"] == str(uid)]
            if "processing_status" in qs:
                st = qs["processing_status"][0].removeprefix("eq.")
                rows = [r for r in rows if r["processing_status"] == st]
            return httpx.Response(200, json=rows)

        return httpx.Response(404, json={"message": "Not found"})


@pytest.fixture(scope="module")
def shared_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder.get_instance()


@pytest.fixture()
def vis_env(shared_embedder: SentenceTransformerEmbedder):
    temp_dir = tempfile.mkdtemp(prefix="retrieva_p4m4_")
    priv = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(priv.public_key()))
    jwk["kid"] = "p4m4-ec-1"
    jwk["use"] = "sig"
    jwk["alg"] = "ES256"

    now = int(time.time())
    token_a = jwt.encode(
        {
            "sub": str(USER_A_ID),
            "aud": "authenticated",
            "iss": f"{SUPABASE_URL}/auth/v1",
            "role": "authenticated",
            "iat": now - 5,
            "exp": now + 3600,
        },
        priv,
        algorithm="ES256",
        headers={"kid": "p4m4-ec-1"},
    )
    token_b = jwt.encode(
        {
            "sub": str(USER_B_ID),
            "aud": "authenticated",
            "iss": f"{SUPABASE_URL}/auth/v1",
            "role": "authenticated",
            "iat": now - 5,
            "exp": now + 3600,
        },
        priv,
        algorithm="ES256",
        headers={"kid": "p4m4-ec-1"},
    )

    sb = VisibilitySupabaseMock({token_a: USER_A_ID, token_b: USER_B_ID})
    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(sb.handle))
    store = ChromaVectorStore(collection_name="p4m4_col", persist_directory=temp_dir)
    bm25 = TenantBM25Manager(vector_store=store)
    retriever = Retriever(
        dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
        tenant_bm25_manager=bm25,
        vector_store=store,
    )

    mock_llm = MagicMock(spec=OpenAIClient)
    mock_llm.model = "gpt-4o-mini"
    mock_llm.generate.return_value = {
        "content": json.dumps(
            {"answer": "Grounded answer [Source 1].", "insufficient_context": False, "citations": []}
        ),
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
    }

    def _stream_gen(*args, **kwargs):
        yield json.dumps(
            {"answer": "Grounded stream [Source 1].", "insufficient_context": False, "citations": []}
        )

    mock_llm.generate_stream.side_effect = _stream_gen
    rag_svc = RAGService(retriever=retriever, llm_client=mock_llm)

    settings = Settings(
        APP_ENV="development",
        AUTH_ENABLED=True,
        SUPABASE_URL=SUPABASE_URL,
        SUPABASE_ANON_KEY=ANON_KEY,
    )

    clear_jwks_client_cache()
    prev_auth = app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_rag_service] = lambda: rag_svc

    def seed_doc(owner_id: UUID, status: str, keyword: str, extra_table: bool = False) -> str:
        doc_id = str(uuid4())
        sb.documents[doc_id] = {
            "id": doc_id,
            "owner_id": str(owner_id),
            "processing_status": status,
        }
        chunks: List[DocumentChunk] = []
        if extra_table:
            t1 = f"| # | Item | Note |\n| --- | --- | --- |\n| 1 | {keyword} | row1 |"
            t2 = f"| 2 | {keyword} continuation | row2 |"
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_p1_c0",
                    document_id=doc_id,
                    filename=f"{keyword}.pdf",
                    page_number=1,
                    page_references=[1],
                    chunk_index=0,
                    chunk_text=t1,
                    start_offset=0,
                    end_offset=len(t1),
                    char_count=len(t1),
                    word_count=len(t1.split()),
                )
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_p1_c1",
                    document_id=doc_id,
                    filename=f"{keyword}.pdf",
                    page_number=1,
                    page_references=[1],
                    chunk_index=1,
                    chunk_text=t2,
                    start_offset=0,
                    end_offset=len(t2),
                    char_count=len(t2),
                    word_count=len(t2.split()),
                )
            )
        else:
            txt = f"Authorized evidence containing unique term {keyword}."
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_p1_c0",
                    document_id=doc_id,
                    filename=f"{keyword}.txt",
                    page_number=1,
                    page_references=[1],
                    chunk_index=0,
                    chunk_text=txt,
                    start_offset=0,
                    end_offset=len(txt),
                    char_count=len(txt),
                    word_count=len(txt.split()),
                )
            )
        embs = shared_embedder.embed_batch([c.chunk_text for c in chunks])
        store.add_chunks(chunks, embs, owner_id, content_hash=uuid4().hex + uuid4().hex)
        bm25.add_chunks(owner_id, chunks)
        return doc_id

    with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}), patch(
        "backend.app.core.supabase_client.get_supabase_http_client",
        return_value=mock_http,
    ):
        with TestClient(app) as client:
            yield {
                "client": client,
                "token_a": token_a,
                "token_b": token_b,
                "sb": sb,
                "store": store,
                "bm25": bm25,
                "retriever": retriever,
                "rag": rag_svc,
                "seed_doc": seed_doc,
                "temp_dir": temp_dir,
                "embedder": shared_embedder,
            }

    app.dependency_overrides.clear()
    if prev_auth is not None:
        app.dependency_overrides[get_current_user] = prev_auth
    clear_jwks_client_cache()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestRetrievalVisibilityHardening:
    """Tests enforcing the ready-only authorized retrieval invariant."""

    def test_status_visibility_parity_across_dense_bm25_hybrid_and_answer(self, vis_env):
        client: TestClient = vis_env["client"]
        tok_a = vis_env["token_a"]
        seed_doc = vis_env["seed_doc"]

        id_up = seed_doc(USER_A_ID, "uploading", "alpha_uploading_only")
        id_idx = seed_doc(USER_A_ID, "indexing", "beta_indexing_only")
        id_fail = seed_doc(USER_A_ID, "failed", "gamma_failed_only")
        id_del = seed_doc(USER_A_ID, "deleting", "delta_deleting_only")
        ready_id = seed_doc(USER_A_ID, "ready", "omega_ready_only")
        blocked_ids = {id_up, id_idx, id_fail, id_del}

        for blocked_kw in ("alpha_uploading_only", "beta_indexing_only", "gamma_failed_only", "delta_deleting_only"):
            for mode in ("dense", "bm25", "hybrid"):
                r = client.post(
                    "/api/v1/retrieval/search",
                    headers={"Authorization": f"Bearer {tok_a}"},
                    json={"query": blocked_kw, "config": {"mode": mode, "top_k": 5}},
                )
                assert r.status_code == 200
                returned_ids = {c["document_id"] for c in r.json()["chunks"]}
                assert not (returned_ids & blocked_ids), f"{blocked_kw} leaked in mode={mode}: {returned_ids}"
                if mode == "bm25":
                    assert r.json()["returned_chunks_count"] == 0

            # Explicit document_ids_filter targeting a non-ready doc must return 0 chunks & insufficient_context=True
            for bid in blocked_ids:
                ans = client.post(
                    "/api/v1/answer",
                    headers={"Authorization": f"Bearer {tok_a}"},
                    json={
                        "query": f"Explain {blocked_kw}",
                        "document_ids": [bid],
                        "retrieval_config": {"mode": "hybrid", "top_k": 5},
                    },
                )
                assert ans.status_code == 200
                assert ans.json()["insufficient_context"] is True
                assert ans.json()["citations"] == []

        # Ready document IS searchable across all 3 modes + /answer
        for mode in ("dense", "bm25", "hybrid"):
            ok_r = client.post(
                "/api/v1/retrieval/search",
                headers={"Authorization": f"Bearer {tok_a}"},
                json={"query": "omega_ready_only", "config": {"mode": mode, "top_k": 5}},
            )
            assert ok_r.status_code == 200
            assert ok_r.json()["returned_chunks_count"] == 1
            assert ok_r.json()["chunks"][0]["document_id"] == ready_id

        ok_ans = client.post(
            "/api/v1/answer",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={"query": "Explain omega_ready_only", "retrieval_config": {"mode": "hybrid", "top_k": 5}},
        )
        assert ok_ans.status_code == 200
        assert ok_ans.json()["insufficient_context"] is False

    def test_concurrent_ingestion_commit_and_deletion_visibility_transitions(self, vis_env):
        client: TestClient = vis_env["client"]
        tok_a = vis_env["token_a"]
        sb: VisibilitySupabaseMock = vis_env["sb"]
        seed_doc = vis_env["seed_doc"]

        # Simulate ChromaDB + BM25 populated while DB is still 'indexing'
        doc_id = seed_doc(USER_A_ID, "indexing", "transition_quartz")
        assert client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={"query": "transition_quartz", "mode": "hybrid", "top_k": 5},
        ).json()["returned_chunks_count"] == 0

        # Commit DB to 'ready' -> immediately visible
        sb.documents[doc_id]["processing_status"] = "ready"
        assert client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={"query": "transition_quartz", "mode": "hybrid", "top_k": 5},
        ).json()["returned_chunks_count"] == 1

        # Begin deletion ('deleting') while Chroma/BM25 still hold stale chunks -> immediately hidden
        sb.documents[doc_id]["processing_status"] = "deleting"
        assert client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={"query": "transition_quartz", "mode": "bm25", "top_k": 5},
        ).json()["returned_chunks_count"] == 0

    def test_stale_bm25_cache_and_table_continuation_visibility(self, vis_env):
        client: TestClient = vis_env["client"]
        tok_a = vis_env["token_a"]
        sb: VisibilitySupabaseMock = vis_env["sb"]
        bm25: TenantBM25Manager = vis_env["bm25"]
        retriever: Retriever = vis_env["retriever"]
        seed_doc = vis_env["seed_doc"]

        # Inject stale chunk directly into BM25 cache for a non-ready doc
        ghost_id = str(uuid4())
        bm25.add_chunks(
            USER_A_ID,
            [{"chunk_id": f"{ghost_id}_c0", "document_id": ghost_id, "filename": "ghost.txt", "chunk_text": "phantom_cobalt_stale"}],
        )
        assert client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={"query": "phantom_cobalt_stale", "mode": "bm25", "top_k": 5},
        ).json()["returned_chunks_count"] == 0

        # Table continuation expansion only expands ready documents
        ready_tbl_id = seed_doc(USER_A_ID, "ready", "ready_table_topaz", extra_table=True)
        deleting_tbl_id = seed_doc(USER_A_ID, "deleting", "del_table_onyx", extra_table=True)

        # Direct expand_table_continuations with authorized_ready_doc_ids excludes deleting_tbl_id
        from backend.app.schemas.retrieval import RetrievedChunk
        fake_chunks = [
            RetrievedChunk(
                chunk_id=f"{deleting_tbl_id}_p1_c0",
                document_id=deleting_tbl_id,
                filename="del.pdf",
                page_number=1,
                page_references=[1],
                chunk_text="| 1 | del_table_onyx | row1 |",
                start_offset=0,
                end_offset=30,
                rank=1,
                metadata={"owner_id": str(USER_A_ID)},
            )
        ]
        expanded = retriever.expand_table_continuations(
            fake_chunks,
            owner_id=USER_A_ID,
            authorized_ready_doc_ids={ready_tbl_id},
        )
        assert expanded == []

    def test_supabase_status_check_failure_fails_closed_and_cross_tenant_blocked(self, vis_env):
        client: TestClient = vis_env["client"]
        tok_a = vis_env["token_a"]
        tok_b = vis_env["token_b"]
        sb: VisibilitySupabaseMock = vis_env["sb"]
        seed_doc = vis_env["seed_doc"]

        b_doc_id = seed_doc(USER_B_ID, "ready", "tenant_b_ruby")

        # User A cannot retrieve User B's ready document even with explicit document_ids_filter
        cross = client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={
                "query": "tenant_b_ruby",
                "mode": "hybrid",
                "config": {"document_ids_filter": [b_doc_id], "top_k": 5},
            },
        )
        assert cross.status_code == 200
        assert cross.json()["returned_chunks_count"] == 0

        # Supabase outage -> fail closed (503) on /retrieval/search, /answer, /answer/stream
        sb.fail_select = True
        try:
            r_fail = client.post(
                "/api/v1/retrieval/search",
                headers={"Authorization": f"Bearer {tok_b}"},
                json={"query": "tenant_b_ruby", "mode": "hybrid", "config": {"top_k": 5}},
            )
            assert r_fail.status_code in {502, 503}

            a_fail = client.post(
                "/api/v1/answer",
                headers={"Authorization": f"Bearer {tok_b}"},
                json={"query": "tenant_b_ruby"},
            )
            assert a_fail.status_code in {502, 503}

            s_fail = client.post(
                "/api/v1/answer/stream",
                headers={"Authorization": f"Bearer {tok_b}"},
                json={"query": "tenant_b_ruby"},
            )
            assert s_fail.status_code in {502, 503}
        finally:
            sb.fail_select = False

    def test_visibility_preserved_after_restart_and_bm25_reconstruction(self, vis_env):
        client: TestClient = vis_env["client"]
        tok_a = vis_env["token_a"]
        seed_doc = vis_env["seed_doc"]

        seed_doc(USER_A_ID, "failed", "restart_hidden_beryl")
        ready_id = seed_doc(USER_A_ID, "ready", "restart_visible_garnet")

        # Simulate restart with fresh ChromaVectorStore, TenantBM25Manager, Retriever
        new_store = ChromaVectorStore(collection_name="p4m4_col", persist_directory=vis_env["temp_dir"])
        new_bm25 = TenantBM25Manager(vector_store=new_store)
        new_retriever = Retriever(
            dense_retriever=DenseRetriever(embedder=vis_env["embedder"], vector_store=new_store),
            tenant_bm25_manager=new_bm25,
            vector_store=new_store,
        )
        app.dependency_overrides[get_retriever] = lambda: new_retriever

        hidden = client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={"query": "restart_hidden_beryl", "mode": "bm25", "top_k": 5},
        )
        assert hidden.status_code == 200
        assert hidden.json()["returned_chunks_count"] == 0

        visible = client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={"query": "restart_visible_garnet", "mode": "bm25", "top_k": 5},
        )
        assert visible.status_code == 200
        assert visible.json()["returned_chunks_count"] == 1
        assert visible.json()["chunks"][0]["document_id"] == ready_id
