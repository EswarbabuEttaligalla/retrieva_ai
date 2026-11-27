"""Milestone 3.3 security, isolation, failure-handling, and Storage tests."""

import io
import json
import logging
import math
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, unquote
from uuid import UUID, uuid4

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from backend.app.core.auth import clear_jwks_client_cache
from backend.app.core.config import Settings, get_settings
from backend.app.core.supabase_client import (
    SupabaseClient,
    SupabaseConflictError,
    SupabaseForbiddenError,
    SupabaseNotFoundError,
)
from backend.app.document_processing.chunker import DocumentChunk
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.main import app, lifespan
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.storage_repository import SupabaseStorageRepository
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.services.ingestion_service import IngestionService, get_ingestion_service
from backend.app.vector_store.client import reset_client_cache
from backend.app.vector_store.store import ChromaVectorStore, TenantIsolationError

USER_A_ID = UUID("10000000-0000-4000-8000-00000000000a")
USER_B_ID = UUID("10000000-0000-4000-8000-00000000000b")
ANON_KEY = "test-public-anon-key"
SUPABASE_URL = "https://zblfcosunplvnnwjfyjz.supabase.co"


def _make_ec_keypair(kid: str = "m33-ec-1") -> Tuple[Any, Dict[str, Any]]:
    priv = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(priv.public_key()))
    jwk["kid"] = kid
    jwk["use"] = "sig"
    jwk["alg"] = "ES256"
    return priv, jwk


def _issue_jwt(priv: Any, user_id: UUID, kid: str = "m33-ec-1") -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": str(user_id),
            "aud": "authenticated",
            "iss": f"{SUPABASE_URL}/auth/v1",
            "role": "authenticated",
            "email": f"{user_id}@example.test",
            "iat": now - 5,
            "exp": now + 3600,
        },
        priv,
        algorithm="ES256",
        headers={"kid": kid},
    )


class MockSupabaseMultiTenantEngine:
    """In-memory multi-tenant simulator for Supabase PostgREST + private Storage RLS."""

    def __init__(self, token_to_user: Dict[str, UUID]) -> None:
        self.token_to_user = token_to_user
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.storage_objects: Dict[str, Dict[str, Any]] = {}

    def _auth_uid(self, request: httpx.Request) -> Optional[UUID]:
        if request.headers.get("apikey") != ANON_KEY:
            return None
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return None
        return self.token_to_user.get(auth[7:])

    def handle(self, request: httpx.Request) -> httpx.Response:
        uid = self._auth_uid(request)
        if uid is None:
            return httpx.Response(401, json={"message": "Unauthorized"})

        path = unquote(request.url.path)
        now = datetime.now(timezone.utc).isoformat()
        qs = parse_qs(request.url.query.decode("utf-8")) if request.url.query else {}

        # ---------------- Storage Bucket RLS ----------------
        if path.startswith("/storage/v1/object/info/authenticated/retrieva-user-documents/"):
            obj_path = path.split("/retrieva-user-documents/", 1)[1]
            if obj_path.split("/")[0] != str(uid):
                return httpx.Response(403, json={"message": "new row violates row-level security policy"})
            obj = self.storage_objects.get(obj_path)
            if not obj:
                return httpx.Response(404, json={"message": "Object not found"})
            return httpx.Response(200, json={"name": obj_path, "size": len(obj["bytes"]), "contentType": obj["content_type"]})

        if path.startswith("/storage/v1/object/authenticated/retrieva-user-documents/"):
            obj_path = path.split("/retrieva-user-documents/", 1)[1]
            if obj_path.split("/")[0] != str(uid):
                return httpx.Response(403, json={"message": "Access denied by storage RLS"})
            obj = self.storage_objects.get(obj_path)
            if not obj:
                return httpx.Response(404, json={"message": "Object not found"})
            return httpx.Response(200, content=obj["bytes"])

        if path.startswith("/storage/v1/object/retrieva-user-documents/"):
            obj_path = path.split("/retrieva-user-documents/", 1)[1]
            if obj_path.split("/")[0] != str(uid):
                return httpx.Response(403, json={"message": "Access denied by storage RLS"})
            if request.method == "POST":
                if obj_path in self.storage_objects and request.headers.get("x-upsert") != "true":
                    return httpx.Response(409, json={"message": "The resource already exists"})
                self.storage_objects[obj_path] = {
                    "bytes": request.content,
                    "content_type": request.headers.get("content-type"),
                }
                return httpx.Response(200, json={"Key": f"retrieva-user-documents/{obj_path}"})
            if request.method == "DELETE":
                if obj_path not in self.storage_objects:
                    return httpx.Response(404, json={"message": "Object not found"})
                del self.storage_objects[obj_path]
                return httpx.Response(200, json={"message": "Deleted"})

        # ---------------- PostgREST Documents + RPCs ----------------
        body = json.loads(request.content.decode("utf-8")) if request.content else {}

        if path == "/rest/v1/rpc/create_user_document" and request.method == "POST":
            doc_id = body.get("p_document_id") or str(uuid4())
            storage_path = body["p_storage_path"]
            if storage_path.split("/")[0] != str(uid):
                return httpx.Response(400, json={"code": "23514", "message": "documents_storage_path_scoped"})
            for existing in self.documents.values():
                if existing["owner_id"] == str(uid) and existing["content_hash"] == body["p_content_hash"]:
                    return httpx.Response(409, json={"code": "23505", "message": "documents_owner_hash_unique"})
            row = {
                "id": doc_id,
                "owner_id": str(uid),
                "storage_path": storage_path,
                "filename": body["p_filename"],
                "file_extension": body["p_file_extension"],
                "mime_type": body["p_mime_type"],
                "content_hash": body["p_content_hash"],
                "file_size_bytes": body["p_file_size_bytes"],
                "processing_status": body.get("p_processing_status", "pending"),
                "embedding_model": body.get("p_embedding_model"),
                "total_pages": 0,
                "total_chunks": 0,
                "error_message": None,
                "created_at": now,
                "updated_at": now,
            }
            self.documents[doc_id] = row
            return httpx.Response(200, json=row)

        if path == "/rest/v1/rpc/update_user_document_status" and request.method == "POST":
            doc_id = body["p_document_id"]
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != str(uid):
                return httpx.Response(200, json=[])
            row["processing_status"] = body["p_processing_status"]
            if body.get("p_total_pages") is not None:
                row["total_pages"] = body["p_total_pages"]
            if body.get("p_total_chunks") is not None:
                row["total_chunks"] = body["p_total_chunks"]
            if body.get("p_embedding_model") is not None:
                row["embedding_model"] = body["p_embedding_model"]
            row["error_message"] = None if row["processing_status"] == "ready" else body.get("p_error_message")
            row["updated_at"] = now
            return httpx.Response(200, json=[row])

        if path == "/rest/v1/rpc/delete_user_document" and request.method == "POST":
            doc_id = body["p_document_id"]
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != str(uid):
                return httpx.Response(200, json=[])
            del self.documents[doc_id]
            return httpx.Response(200, json=[row])

        if path == "/rest/v1/documents" and request.method == "GET":
            rows = [r for r in self.documents.values() if r["owner_id"] == str(uid)]
            if "id" in qs:
                tid = qs["id"][0].removeprefix("eq.")
                rows = [r for r in rows if r["id"] == tid]
            if "content_hash" in qs:
                thash = qs["content_hash"][0].removeprefix("eq.")
                rows = [r for r in rows if r["content_hash"] == thash]
            return httpx.Response(200, json=rows)

        return httpx.Response(404, json={"message": "Not found"})


@pytest.fixture(autouse=True)
def _reset_jwks():
    clear_jwks_client_cache()
    yield
    clear_jwks_client_cache()


class TestSupabaseStorageIsolation:
    """Tests for SupabaseStorageRepository validation, path scoping, and RLS enforcement."""

    @pytest.mark.asyncio
    async def test_storage_upload_download_metadata_delete_and_cross_user_denial(self):
        engine = MockSupabaseMultiTenantEngine({"jwt-a": USER_A_ID, "jwt-b": USER_B_ID})
        async with httpx.AsyncClient(transport=httpx.MockTransport(engine.handle)) as pool:
            client_a = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-a", USER_A_ID, http_client=pool)
            client_b = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-b", USER_B_ID, http_client=pool)
            storage_a = SupabaseStorageRepository(client_a)
            storage_b = SupabaseStorageRepository(client_b)

            doc_a_id = uuid4()
            payload = b"%PDF-1.4\nSample private PDF bytes"
            uploaded = await storage_a.upload_document(
                owner_id=USER_A_ID,
                document_id=doc_a_id,
                filename="../../secret notes?.pdf",
                file_bytes=payload,
                declared_mime="application/x-pdf",
            )
            expected_prefix = f"{USER_A_ID}/{doc_a_id}/"
            assert uploaded["storage_path"].startswith(expected_prefix)
            assert uploaded["mime_type"] == "application/pdf"
            assert ".." not in uploaded["storage_path"]

            # Duplicate upload without upsert raises conflict (409)
            with pytest.raises(SupabaseConflictError):
                await storage_a.upload_document(
                    owner_id=USER_A_ID,
                    document_id=doc_a_id,
                    filename="../../secret notes?.pdf",
                    file_bytes=payload,
                    upsert=False,
                )

            # Owner A can download & inspect metadata
            assert await storage_a.download_document(USER_A_ID, uploaded["storage_path"]) == payload
            meta = await storage_a.get_document_metadata(USER_A_ID, uploaded["storage_path"])
            assert meta is not None and meta["size"] == len(payload)

            # User B cannot download, overwrite, or delete User A's storage object
            with pytest.raises(SupabaseForbiddenError):
                await storage_b.download_document(USER_B_ID, uploaded["storage_path"])
            with pytest.raises(SupabaseForbiddenError):
                await storage_b.delete_document(USER_B_ID, uploaded["storage_path"])
            with pytest.raises(SupabaseForbiddenError):
                await storage_b.get_document_metadata(USER_B_ID, uploaded["storage_path"])
            # Even if User B crafts raw Storage HTTP call bypassing repo path check, Storage RLS returns 403
            with pytest.raises(SupabaseForbiddenError):
                await client_b.request_raw(
                    "POST",
                    f"{SUPABASE_URL}/storage/v1/object/retrieva-user-documents/{uploaded['storage_path']}",
                    content=b"%PDF-overwrite",
                    extra_headers={"Content-Type": "application/pdf", "x-upsert": "true"},
                )

            # Missing object behaviors
            missing_path = f"{USER_A_ID}/{uuid4()}/missing.txt"
            with pytest.raises(SupabaseNotFoundError):
                await storage_a.download_document(USER_A_ID, missing_path)
            assert await storage_a.delete_document(USER_A_ID, missing_path) is False
            assert await storage_a.get_document_metadata(USER_A_ID, missing_path) is None

            # Owner A deletes cleanly; retry delete is idempotent
            assert await storage_a.delete_document(USER_A_ID, uploaded["storage_path"]) is True
            assert await storage_a.delete_document(USER_A_ID, uploaded["storage_path"]) is False


class TestChromaTenantIsolation:
    """Tests for mandatory ChromaDB owner filtering and post-retrieval assertions."""

    @pytest.fixture
    def isolated_store(self):
        temp_dir = tempfile.mkdtemp(prefix="retrieva_m33_chroma_")
        store = ChromaVectorStore(collection_name="m33_test_col", persist_directory=temp_dir)
        yield store
        reset_client_cache()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _chunk(doc_id: str, idx: int, text: str) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=f"{doc_id}_p1_c{idx}",
            document_id=doc_id,
            filename="doc.txt",
            page_number=1,
            page_references=[1],
            chunk_index=idx,
            chunk_text=text,
            start_offset=0,
            end_offset=len(text),
            char_count=len(text),
            word_count=len(text.split()),
        )

    def test_mandatory_owner_filter_and_cross_tenant_block(self, isolated_store, caplog):
        embedder = SentenceTransformerEmbedder.get_instance()
        doc_a = str(uuid4())
        doc_b = str(uuid4())
        chunk_a = self._chunk(doc_a, 0, "AlphaProject proprietary financial projection 2027.")
        chunk_b = self._chunk(doc_b, 0, "BetaProject confidential merger roadmap 2027.")

        emb_a = embedder.embed_batch([chunk_a.chunk_text])
        emb_b = embedder.embed_batch([chunk_b.chunk_text])

        isolated_store.add_chunks([chunk_a], emb_a, owner_id=USER_A_ID, content_hash="1" * 64)
        isolated_store.add_chunks([chunk_b], emb_b, owner_id=USER_B_ID, content_hash="1" * 64)

        # Identical content_hash allowed across distinct owners, distinct UUIDs
        assert isolated_store.has_document(doc_a, owner_id=USER_A_ID) is True
        assert isolated_store.has_document(doc_b, owner_id=USER_A_ID) is False
        assert isolated_store.get_document_chunks(doc_b, owner_id=USER_A_ID) == []
        assert isolated_store.delete_document(doc_b, owner_id=USER_A_ID) == 0
        assert isolated_store.has_document(doc_b, owner_id=USER_B_ID) is True

        # Malformed owner_id is rejected
        with pytest.raises(ValueError):
            isolated_store.query_similar(emb_a[0], owner_id="not-a-uuid", top_k=5)

        # Querying as User A returns only User A's chunks, even when filtering for User B's doc UUID
        res_a = isolated_store.query_similar(emb_b[0], owner_id=USER_A_ID, top_k=5)
        assert all(m["owner_id"] == str(USER_A_ID) for m in res_a["metadatas"][0])

        res_cross_doc = isolated_store.query_similar(
            emb_b[0],
            owner_id=USER_A_ID,
            top_k=5,
            document_id_filter=doc_b,
        )
        assert res_cross_doc["ids"] == [[]]

        # Collection statistics without owner_id never expose document IDs
        sys_stats = isolated_store.get_statistics()
        assert "unique_document_ids" not in sys_stats
        owner_a_stats = isolated_store.get_statistics(owner_id=USER_A_ID)
        assert owner_a_stats["unique_document_ids"] == [doc_a]

        # Defensive post-retrieval assertion blocks leaked foreign chunk & avoids logging chunk text
        secret_text = "TOP_SECRET_FOREIGN_CHUNK_TEXT"
        with patch.object(
            isolated_store.collection,
            "query",
            return_value={
                "ids": [["foreign_chunk_0"]],
                "documents": [[secret_text]],
                "metadatas": [[{"owner_id": str(USER_B_ID), "document_id": doc_b}]],
                "distances": [[0.1]],
            },
        ):
            with caplog.at_level(logging.ERROR, logger="retrieva_ai"):
                with pytest.raises(TenantIsolationError):
                    isolated_store.query_similar(emb_a[0], owner_id=USER_A_ID, top_k=5)
                assert secret_text not in caplog.text


class TestTwoUserEndToEndAndFailureRecovery:
    """Two-user API isolation, startup safety, and failure rollback tests."""

    @pytest.fixture
    def multi_tenant_app(self):
        temp_dir = tempfile.mkdtemp(prefix="retrieva_m33_e2e_")
        priv, jwk = _make_ec_keypair("m33-e2e")
        token_a = _issue_jwt(priv, USER_A_ID, kid="m33-e2e")
        token_b = _issue_jwt(priv, USER_B_ID, kid="m33-e2e")

        engine = MockSupabaseMultiTenantEngine({token_a: USER_A_ID, token_b: USER_B_ID})
        mock_http = httpx.AsyncClient(transport=httpx.MockTransport(engine.handle))

        embedder = SentenceTransformerEmbedder.get_instance()
        vector_store = ChromaVectorStore(collection_name="m33_e2e_col", persist_directory=temp_dir)
        ingestion_svc = IngestionService(embedder=embedder, vector_store=vector_store)
        retriever = Retriever(
            dense_retriever=DenseRetriever(embedder=embedder, vector_store=vector_store),
            vector_store=vector_store,
        )
        settings = Settings(
            APP_ENV="development",
            SUPABASE_URL=SUPABASE_URL,
            SUPABASE_ANON_KEY=ANON_KEY,
        )

        prev_settings = app.dependency_overrides.get(get_settings)
        prev_ing = app.dependency_overrides.get(get_ingestion_service)
        prev_ret = app.dependency_overrides.get(get_retriever)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_ingestion_service] = lambda: ingestion_svc
        app.dependency_overrides[get_retriever] = lambda: retriever

        try:
            with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}), patch(
                "backend.app.core.supabase_client.get_supabase_http_client",
                return_value=mock_http,
            ):
                with TestClient(app) as client:
                    yield client, token_a, token_b, engine, vector_store, ingestion_svc
        finally:
            for key, prev in [
                (get_settings, prev_settings),
                (get_ingestion_service, prev_ing),
                (get_retriever, prev_ret),
            ]:
                if prev is None:
                    app.dependency_overrides.pop(key, None)
                else:
                    app.dependency_overrides[key] = prev
            reset_client_cache()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_two_user_upload_identical_file_retrieve_and_delete_isolation(self, multi_tenant_app):
        client, token_a, token_b, engine, vector_store, _ = multi_tenant_app
        identical_bytes = b"Shared standard specification document about quantum cryptography."

        res_a = client.post(
            "/api/v1/documents/upload",
            headers={"Authorization": f"Bearer {token_a}"},
            files={"file": ("shared_spec.txt", io.BytesIO(identical_bytes), "text/plain")},
        )
        assert res_a.status_code == 200
        doc_a_id = res_a.json()["document_id"]
        UUID(doc_a_id)

        # User B uploads the exact same file bytes -> gets a distinct UUID, NOT duplicate error
        res_b = client.post(
            "/api/v1/documents/upload",
            headers={"Authorization": f"Bearer {token_b}"},
            files={"file": ("shared_spec_copy.txt", io.BytesIO(identical_bytes), "text/plain")},
        )
        assert res_b.status_code == 200
        doc_b_id = res_b.json()["document_id"]
        UUID(doc_b_id)
        assert doc_b_id != doc_a_id
        assert res_b.json()["is_duplicate"] is False

        # Listing is isolated per user
        list_a = client.get("/api/v1/documents", headers={"Authorization": f"Bearer {token_a}"}).json()
        list_b = client.get("/api/v1/documents", headers={"Authorization": f"Bearer {token_b}"}).json()
        assert [d["document_id"] for d in list_a["documents"]] == [doc_a_id]
        assert [d["document_id"] for d in list_b["documents"]] == [doc_b_id]

        # Dense retrieval returns only caller's chunks; filtering by foreign doc UUID returns 0 chunks
        search_a = client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"query": "quantum cryptography", "mode": "dense", "top_k": 5},
        )
        assert search_a.status_code == 200
        chunks_a = search_a.json()["chunks"]
        assert len(chunks_a) >= 1
        assert all(c["document_id"] == doc_a_id for c in chunks_a)

        cross_search = client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"query": "quantum cryptography", "mode": "dense", "top_k": 5, "document_id_filter": doc_b_id},
        )
        assert cross_search.status_code == 200
        assert cross_search.json()["chunks"] == []

        # User A cannot delete User B's document UUID (returns 404; B's doc remains intact)
        del_cross = client.delete(
            f"/api/v1/documents/{doc_b_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert del_cross.status_code == 404
        assert vector_store.has_document(doc_b_id, owner_id=USER_B_ID) is True
        assert doc_b_id in engine.documents

        # User A deletes own document -> User B's identical document remains searchable
        del_a = client.delete(
            f"/api/v1/documents/{doc_a_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert del_a.status_code == 200
        assert vector_store.has_document(doc_a_id, owner_id=USER_A_ID) is False
        assert vector_store.has_document(doc_b_id, owner_id=USER_B_ID) is True

    def test_ingestion_and_deletion_failure_cleanup_idempotency(self, multi_tenant_app):
        client, token_a, _, engine, vector_store, ingestion_svc = multi_tenant_app

        # 1. Storage upload failure does not leave an accessible 'ready' document
        with patch.object(
            SupabaseStorageRepository,
            "upload_document",
            side_effect=RuntimeError("Simulated Storage outage"),
        ):
            fail_storage = client.post(
                "/api/v1/documents/upload",
                headers={"Authorization": f"Bearer {token_a}"},
                files={"file": ("storage_fail.txt", io.BytesIO(b"Storage failure content."), "text/plain")},
            )
            assert fail_storage.status_code == 500
        assert client.get("/api/v1/documents", headers={"Authorization": f"Bearer {token_a}"}).json()["total"] == 0
        assert vector_store.count_for_owner(USER_A_ID) == 0

        # 2. Embedding failure leaves zero vectors and marks 'failed'
        with patch.object(
            ingestion_svc.embedder,
            "embed_batch",
            side_effect=RuntimeError("Embedding failure"),
        ):
            fail_embed = client.post(
                "/api/v1/documents/upload",
                headers={"Authorization": f"Bearer {token_a}"},
                files={"file": ("embed_fail.txt", io.BytesIO(b"Embedding failure content."), "text/plain")},
            )
            assert fail_embed.status_code == 500
        assert vector_store.count_for_owner(USER_A_ID) == 0
        assert any(r["processing_status"] == "failed" for r in engine.documents.values())
        assert client.get("/api/v1/documents", headers={"Authorization": f"Bearer {token_a}"}).json()["total"] == 0

        # 3. ChromaDB insertion failure marks 'failed' and keeps collection clean
        with patch.object(
            vector_store,
            "add_chunks",
            side_effect=RuntimeError("ChromaDB failure"),
        ):
            fail_chroma = client.post(
                "/api/v1/documents/upload",
                headers={"Authorization": f"Bearer {token_a}"},
                files={"file": ("chroma_fail.txt", io.BytesIO(b"Chroma failure content."), "text/plain")},
            )
            assert fail_chroma.status_code == 500
        assert vector_store.count_for_owner(USER_A_ID) == 0

        # 4. Successful upload followed by Storage deletion failure must NOT report success & must allow idempotent retry
        ok_res = client.post(
            "/api/v1/documents/upload",
            headers={"Authorization": f"Bearer {token_a}"},
            files={"file": ("retry_delete.txt", io.BytesIO(b"Retryable delete content."), "text/plain")},
        )
        assert ok_res.status_code == 200
        doc_id = ok_res.json()["document_id"]

        with patch.object(
            SupabaseStorageRepository,
            "delete_document",
            side_effect=RuntimeError("Transient Storage delete error"),
        ):
            del_fail = client.delete(
                f"/api/v1/documents/{doc_id}",
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert del_fail.status_code == 500

        # Status preserved as 'deleting' and doc is no longer listed as 'ready'
        assert engine.documents[doc_id]["processing_status"] == "deleting"
        assert client.get("/api/v1/documents", headers={"Authorization": f"Bearer {token_a}"}).json()["total"] == 0

        # Idempotent retry succeeds cleanly
        del_retry = client.delete(
            f"/api/v1/documents/{doc_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert del_retry.status_code == 200
        assert doc_id not in engine.documents

    @pytest.mark.asyncio
    async def test_startup_never_loads_unowned_chromadb_chunks(self):
        temp_dir = tempfile.mkdtemp(prefix="retrieva_m33_startup_")
        try:
            store = ChromaVectorStore(collection_name="m33_startup_col", persist_directory=temp_dir)
            embedder = SentenceTransformerEmbedder.get_instance()
            vec = embedder.embed_batch(["Legacy unowned chunk"])[0]
            # Simulate an unowned legacy record directly in ChromaDB
            store.collection.add(
                ids=["legacy_unowned_0"],
                embeddings=[vec],
                documents=["Legacy unowned secret text"],
                metadatas=[{"document_id": "doc_legacy", "filename": "legacy.txt", "page_number": 1}],
            )
            retriever = Retriever(
                dense_retriever=DenseRetriever(embedder=embedder, vector_store=store),
                vector_store=store,
            )
            owner_bm25 = retriever.tenant_bm25_manager.get_or_create_index(USER_A_ID)
            assert owner_bm25.doc_count == 0
            assert retriever.tenant_bm25_manager.get_index(USER_A_ID).doc_count == 0
        finally:
            reset_client_cache()
            shutil.rmtree(temp_dir, ignore_errors=True)
