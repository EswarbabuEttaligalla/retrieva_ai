"""Automated tests for document upload, listing, and deletion API endpoints.

Covers:
- Valid PDF and TXT document uploads
- File size boundary checks (0 bytes, > 10MB)
- Unsupported file extension rejection
- Corrupted PDF magic-byte validation
- Duplicate upload handling
- Document listing and schema adherence
- Document deletion and 404 handling
- Verification that deleted documents are purged from Dense and BM25 retrieval
"""

import codecs
import io
from unittest.mock import patch
import fitz  # PyMuPDF
import pytest
from starlette.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.document_processing.identifier import global_document_registry
from backend.app.retrieval.retriever import get_retriever
from backend.app.schemas.retrieval import RetrievalConfiguration, RetrievalMode, RetrievalRequest
from backend.app.services.ingestion_service import get_ingestion_service


def create_sample_pdf(text: str = "Retrieva AI provides grounded document retrieval.") -> bytes:
    """Create a minimal valid in-memory single-page PDF document."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestDocumentAPIEndpoints:
    """Test suite for /api/v1/documents endpoints."""

    def setup_method(self):
        """Clean up before each test."""
        self.settings = get_settings()
        self.ingestion_service = get_ingestion_service()
        self.retriever = get_retriever()

    def test_upload_valid_txt_success(self, sync_client: TestClient):
        """Verify successful upload, chunking, and metadata indexing for TXT file."""
        content = b"Superconductors exhibit zero electrical resistance below critical temperature."
        files = {"file": ("physics_intro.txt", io.BytesIO(content), "text/plain")}

        response = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data["filename"] == "physics_intro.txt"
        assert data["status"] == "success"
        assert data["is_duplicate"] is False
        assert data["total_pages"] == 1
        assert data["total_chunks"] >= 1
        assert data["file_size_bytes"] == len(content)
        assert data["document_id"].startswith("doc_")

        # Cleanup
        self.ingestion_service.delete_document(data["document_id"])
        self.retriever.sync_bm25_with_vector_store()

    def test_upload_valid_pdf_success(self, sync_client: TestClient):
        """Verify successful upload and indexing for a standard PDF file."""
        pdf_bytes = create_sample_pdf("Neural network transformers utilize self-attention mechanisms.")
        files = {"file": ("transformers_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

        response = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data["filename"] == "transformers_paper.pdf"
        assert data["status"] == "success"
        assert data["is_duplicate"] is False
        assert data["total_pages"] == 1
        assert data["total_chunks"] >= 1
        assert data["document_id"].startswith("doc_")

        # Cleanup
        self.ingestion_service.delete_document(data["document_id"])
        self.retriever.sync_bm25_with_vector_store()

    def test_upload_unsupported_extension_rejected(self, sync_client: TestClient):
        """Verify that files with unsupported extensions (.docx, .png, etc.) return 400 Bad Request."""
        files = {"file": ("notes.docx", io.BytesIO(b"fake word doc content"), "application/octet-stream")}
        response = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)

        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_upload_empty_file_rejected(self, sync_client: TestClient):
        """Verify that zero-byte files are rejected with 400 Bad Request."""
        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        response = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_upload_oversized_file_rejected(self, sync_client: TestClient, monkeypatch):
        """Verify that files exceeding the maximum allowed size are rejected with 413."""
        # Monkeypatch size limit to 500 bytes for fast testing
        monkeypatch.setattr(self.settings, "MAX_UPLOAD_SIZE_BYTES", 500)

        oversized_bytes = b"X" * 600
        files = {"file": ("huge_file.txt", io.BytesIO(oversized_bytes), "text/plain")}
        response = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)

        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"].lower()

    def test_upload_corrupted_pdf_rejected(self, sync_client: TestClient):
        """Verify that corrupted non-PDF bytes bearing a .pdf extension are rejected with 400."""
        corrupted_bytes = b"THIS IS NOT A VALID PDF CONTENT AT ALL"
        files = {"file": ("corrupt.pdf", io.BytesIO(corrupted_bytes), "application/pdf")}

        response = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert response.status_code == 400
        assert "pdf" in response.json()["detail"].lower()

    def test_upload_duplicate_detection(self, sync_client: TestClient):
        """Verify duplicate detection when uploading identical content twice."""
        content = b"Consistent content for duplicate testing across multiple uploads."
        files1 = {"file": ("doc_first.txt", io.BytesIO(content), "text/plain")}
        files2 = {"file": ("doc_second.txt", io.BytesIO(content), "text/plain")}

        res1 = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files1)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["status"] == "success"
        assert data1["is_duplicate"] is False

        res2 = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files2)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["status"] == "duplicate"
        assert data2["is_duplicate"] is True
        assert data2["document_id"] == data1["document_id"]
        assert "already" in data2["message"].lower()

        # Cleanup
        self.ingestion_service.delete_document(data1["document_id"])
        self.retriever.sync_bm25_with_vector_store()

    def test_list_documents(self, sync_client: TestClient):
        """Verify GET /api/v1/documents returns accurate catalog of stored documents."""
        # Clean any preexisting state
        for d in self.ingestion_service.registry.get_all():
            self.ingestion_service.delete_document(d.document_id)

        # Ingest two distinct documents
        content1 = b"First unique document on thermodynamics and entropy."
        content2 = b"Second unique document on algorithmic complexity and graph algorithms."

        sync_client.post(
            f"{self.settings.API_V1_STR}/documents/upload",
            files={"file": ("thermo.txt", io.BytesIO(content1), "text/plain")},
        )
        sync_client.post(
            f"{self.settings.API_V1_STR}/documents/upload",
            files={"file": ("algorithms.txt", io.BytesIO(content2), "text/plain")},
        )

        response = sync_client.get(f"{self.settings.API_V1_STR}/documents")
        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 2
        assert len(data["documents"]) == 2

        filenames = [d["filename"] for d in data["documents"]]
        assert "thermo.txt" in filenames
        assert "algorithms.txt" in filenames

        first_item = data["documents"][0]
        assert "document_id" in first_item
        assert "file_extension" in first_item
        assert "file_size_bytes" in first_item
        assert "total_pages" in first_item
        assert "total_chunks" in first_item
        assert "created_at" in first_item

        # Cleanup
        for d in data["documents"]:
            self.ingestion_service.delete_document(d["document_id"])
        self.retriever.sync_bm25_with_vector_store()

    def test_delete_document_success(self, sync_client: TestClient):
        """Verify DELETE /api/v1/documents/{id} removes document and returns deleted chunk count."""
        content = b"Document to be deleted soon from Retrieva knowledge base."
        res_upload = sync_client.post(
            f"{self.settings.API_V1_STR}/documents/upload",
            files={"file": ("to_delete.txt", io.BytesIO(content), "text/plain")},
        )
        doc_id = res_upload.json()["document_id"]

        del_res = sync_client.delete(f"{self.settings.API_V1_STR}/documents/{doc_id}")
        assert del_res.status_code == 200
        del_data = del_res.json()
        assert del_data["document_id"] == doc_id
        assert del_data["status"] == "deleted"
        assert del_data["deleted_chunks"] >= 1

        # Verify document is no longer listed
        list_res = sync_client.get(f"{self.settings.API_V1_STR}/documents")
        existing_ids = [d["document_id"] for d in list_res.json()["documents"]]
        assert doc_id not in existing_ids

    def test_delete_nonexistent_document_returns_404(self, sync_client: TestClient):
        """Verify DELETE /api/v1/documents/{nonexistent_id} returns 404 Not Found."""
        response = sync_client.delete(f"{self.settings.API_V1_STR}/documents/doc_does_not_exist_999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_deleted_document_excluded_from_retrieval(self, sync_client: TestClient):
        """Verify that after document deletion, its contents are not returned by Dense or BM25 retrieval."""
        unique_term = "XylophonicResonatorKeyphrase987"
        text = f"The {unique_term} is an acoustic engineering marvel producing harmonic tones."

        # Upload
        res = sync_client.post(
            f"{self.settings.API_V1_STR}/documents/upload",
            files={"file": ("acoustics.txt", io.BytesIO(text.encode("utf-8")), "text/plain")},
        )
        doc_id = res.json()["document_id"]

        # Ensure retrieval finds it before deletion
        req_dense = RetrievalRequest(
            query=f"Tell me about the {unique_term}",
            config=RetrievalConfiguration(mode=RetrievalMode.DENSE, top_k=5),
        )
        res_dense = self.retriever.retrieve(req_dense)
        assert any(unique_term in c.chunk_text for c in res_dense.chunks)

        req_bm25 = RetrievalRequest(
            query=unique_term,
            config=RetrievalConfiguration(mode=RetrievalMode.BM25, top_k=5),
        )
        res_bm25 = self.retriever.retrieve(req_bm25)
        assert any(unique_term in c.chunk_text for c in res_bm25.chunks)

        # Delete document via API
        del_res = sync_client.delete(f"{self.settings.API_V1_STR}/documents/{doc_id}")
        assert del_res.status_code == 200

        # Verify retrieval yields 0 chunks for that unique term across both modes
        res_dense_after = self.retriever.retrieve(req_dense)
        assert not any(unique_term in c.chunk_text for c in res_dense_after.chunks)

        res_bm25_after = self.retriever.retrieve(req_bm25)
        assert not any(unique_term in c.chunk_text for c in res_bm25_after.chunks)

    def test_upload_disallowed_mime_type_rejected_415(self, sync_client: TestClient):
        """Verify that files with mismatched or disallowed MIME types return 415 Unsupported Media Type."""
        pdf_bytes = create_sample_pdf("Sample text.")
        # Uploading PDF bytes with an executable MIME type
        files = {"file": ("malicious.pdf", io.BytesIO(pdf_bytes), "application/x-msdownload")}
        response = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)

        assert response.status_code == 415
        assert "unsupported media type" in response.json()["detail"].lower()

    def test_upload_path_traversal_sanitized(self, sync_client: TestClient):
        """Verify that directory traversal attempts (e.g. ../../etc/config.txt) are sanitized to basename."""
        content = b"Safe text content inside simulated path traversal attempt."
        files = {"file": ("../../etc/config.txt", io.BytesIO(content), "text/plain")}

        response = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data["filename"] == "config.txt"
        assert "/" not in data["filename"]
        assert "\\" not in data["filename"]

        # Cleanup
        self.ingestion_service.delete_document(data["document_id"])
        self.retriever.sync_bm25_with_vector_store()

    def test_upload_path_traversal_pure_dots_rejected(self, sync_client: TestClient):
        """Verify that filenames consisting only of path traversal dots (e.g. '..') are rejected."""
        files = {"file": ("..", io.BytesIO(b"content"), "text/plain")}
        response = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert response.status_code == 400
        assert "valid document filename" in response.json()["detail"].lower()

    def test_registry_persistence_failure_rollback(self):
        """Verify that if disk persistence fails, DocumentRegistry rolls back in-memory mutation."""
        from unittest.mock import patch
        from backend.app.document_processing.exceptions import RegistryPersistenceError
        from backend.app.document_processing.identifier import DocumentRegistry
        from backend.app.schemas.document import DocumentMetadata

        registry = DocumentRegistry(persist_path=None)
        meta = DocumentMetadata(
            document_id="doc_rollback_test",
            filename="test.txt",
            file_extension=".txt",
            file_size_bytes=100,
            content_hash="hash_rollback_test_123",
            total_pages=1,
            total_chunks=1,
        )

        with patch.object(registry, "_save", side_effect=RegistryPersistenceError("Simulated disk full")):
            with pytest.raises(RegistryPersistenceError):
                registry.register(meta, raise_on_error=True)

        # In-memory storage must NOT contain the failed document
        assert registry.get_by_hash("hash_rollback_test_123") is None
        assert registry.get_by_id("doc_rollback_test") is None

    def test_upload_ascii_txt_success(self, sync_client: TestClient):
        """Verify upload of standard ASCII text file succeeds."""
        content = b"Ordinary ASCII text describing fundamental algorithms and data structures."
        files = {"file": ("algorithms.txt", io.BytesIO(content), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 200
        doc_id = res.json()["document_id"]
        self.ingestion_service.delete_document(doc_id)
        self.retriever.sync_bm25_with_vector_store()

    def test_upload_multilingual_utf8_txt_success(self, sync_client: TestClient):
        """Verify upload of multilingual UTF-8 text file succeeds."""
        content = (
            "Multilingual documentation: Greek \u03b1\u03b2\u03b3, Arabic \u0645\u0631\u062d\u0628\u0627, Japanese \u3053\u3093\u306b\u3061\u306f, French caf\u00e9, German Z\u00fcrich."
        ).encode("utf-8")
        files = {"file": ("multilingual_doc.txt", io.BytesIO(content), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 200
        doc_id = res.json()["document_id"]
        self.ingestion_service.delete_document(doc_id)
        self.retriever.sync_bm25_with_vector_store()

    def test_upload_utf8_bom_txt_success(self, sync_client: TestClient):
        """Verify upload of UTF-8 file with Byte Order Mark (BOM) succeeds."""
        content = codecs.BOM_UTF8 + "Document starting with UTF-8 byte order mark header.".encode("utf-8")
        files = {"file": ("bom_document.txt", io.BytesIO(content), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 200
        doc_id = res.json()["document_id"]
        self.ingestion_service.delete_document(doc_id)
        self.retriever.sync_bm25_with_vector_store()

    def test_upload_invalid_utf8_binary_payload_rejected_400(self, sync_client: TestClient):
        """Verify that invalid UTF-8 byte sequences are rejected with 400 Bad Request."""
        content = b"Valid ASCII prefix \x80\xFF\xFE\xFD invalid non-utf8 continuation"
        files = {"file": ("invalid_utf8.txt", io.BytesIO(content), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 400
        assert "utf-8" in res.json()["detail"].lower()

    def test_upload_binary_null_bytes_txt_rejected_400(self, sync_client: TestClient):
        """Verify that text files with embedded binary null bytes are rejected with 400 Bad Request."""
        content = b"Some text content\x00with embedded null byte indicating binary stream."
        files = {"file": ("binary_null.txt", io.BytesIO(content), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 400
        assert "null bytes" in res.json()["detail"].lower() or "binary" in res.json()["detail"].lower()

    def test_upload_binary_file_renamed_to_txt_rejected_400(self, sync_client: TestClient):
        """Verify that binary executable/archive files renamed to .txt are rejected."""
        # Simulated ELF executable header
        content = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00>\x00"
        files = {"file": ("compiled_binary.txt", io.BytesIO(content), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 400

    def test_upload_empty_txt_file_rejected_400(self, sync_client: TestClient):
        """Verify that 0-byte TXT uploads are rejected with 400 Bad Request."""
        files = {"file": ("zero_bytes.txt", io.BytesIO(b""), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()

    def test_upload_txt_generic_octet_stream_rejected_415(self, sync_client: TestClient):
        """Verify that application/octet-stream is rejected with 415 for TXT files."""
        content = b"Valid text but generic binary MIME type."
        files = {"file": ("octet_stream.txt", io.BytesIO(content), "application/octet-stream")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 415
        assert "unsupported media type" in res.json()["detail"].lower()

    def test_upload_txt_disallowed_image_mime_rejected_415(self, sync_client: TestClient):
        """Verify that image MIME types are rejected with 415 for TXT files."""
        content = b"Valid text but mismatched image MIME type."
        files = {"file": ("fake_image.txt", io.BytesIO(content), "image/jpeg")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 415
        assert "unsupported media type" in res.json()["detail"].lower()

    def test_delete_recovery_chromadb_succeeds_metadata_fails(self, sync_client: TestClient):
        """Verify multi-store failure recovery when ChromaDB deletion succeeds but metadata persistence fails."""
        content = b"Sample text for delete recovery failure test."
        files = {"file": ("delete_recovery_test.txt", io.BytesIO(content), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 200
        doc_id = res.json()["document_id"]

        with patch.object(self.ingestion_service.registry, "remove", side_effect=Exception("Disk persistence failure")):
            del_res = sync_client.delete(f"{self.settings.API_V1_STR}/documents/{doc_id}")
            assert del_res.status_code == 500
            assert "metadata registry persistence failed" in del_res.json()["detail"].lower()

        # Clean up
        self.ingestion_service.delete_document(doc_id)
        self.retriever.sync_bm25_with_vector_store()

    def test_delete_recovery_metadata_succeeds_bm25_fails(self, sync_client: TestClient):
        """Verify multi-store failure behavior when metadata deletion succeeds but BM25 rebuilding fails."""
        content = b"Sample text for bm25 failure recovery test."
        files = {"file": ("bm25_failure_test.txt", io.BytesIO(content), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 200
        doc_id = res.json()["document_id"]

        with patch.object(self.retriever, "sync_bm25_with_vector_store", side_effect=RuntimeError("Simulated BM25 crash")):
            del_res = sync_client.delete(f"{self.settings.API_V1_STR}/documents/{doc_id}")
            assert del_res.status_code == 500
            assert "bm25 keyword index synchronization failed" in del_res.json()["detail"].lower()

        # Re-sync to restore state
        self.retriever.sync_bm25_with_vector_store()

    def test_restart_after_partial_deletion_reconciliation(self):
        """Verify that reconciliation during restart prunes orphaned metadata records left by partial deletions."""
        from backend.app.schemas.document import DocumentMetadata
        orphan_id = "doc_partial_orphan_999"
        orphan_meta = DocumentMetadata(
            document_id=orphan_id,
            filename="orphan.txt",
            file_extension=".txt",
            file_size_bytes=50,
            content_hash="hash_orphan_999",
            total_pages=1,
            total_chunks=1,
        )
        self.ingestion_service.registry.register(orphan_meta)
        assert self.ingestion_service.registry.get_by_id(orphan_id) is not None

        # Reconcile against vector store (which has 0 chunks for this orphan)
        pruned = self.ingestion_service.registry.reconcile_from_vector_store(
            self.ingestion_service.vector_store, prune_orphans=True
        )
        assert pruned >= 1
        assert self.ingestion_service.registry.get_by_id(orphan_id) is None

    def test_reconciliation_discovers_uncataloged_chromadb_document(self):
        """Verify that reconciliation discovers chunks in ChromaDB missing from DocumentRegistry."""
        import numpy as np
        from backend.app.schemas.document import DocumentChunk
        doc_id = "doc_uncataloged_direct"
        norm_vec = np.zeros(384, dtype=np.float32)
        norm_vec[0] = 1.0

        chunk = DocumentChunk(
            chunk_id=f"{doc_id}_chunk_0",
            document_id=doc_id,
            filename="uncataloged.txt",
            content_hash="hash_uncataloged_direct",
            chunk_index=0,
            page_number=1,
            start_offset=0,
            end_offset=42,
            chunk_text="Direct uncataloged ChromaDB chunk content.",
            char_count=42,
            word_count=5,
        )
        self.ingestion_service.vector_store.add_chunks(
            chunks=[chunk],
            embeddings=[norm_vec.tolist()],
        )

        # Confirm registry does not have it yet
        if self.ingestion_service.registry.get_by_id(doc_id):
            self.ingestion_service.registry.remove(doc_id)

        # Run reconciliation
        reconciled = self.ingestion_service.registry.reconcile_from_vector_store(
            self.ingestion_service.vector_store, prune_orphans=False
        )
        assert reconciled >= 1
        meta = self.ingestion_service.registry.get_by_id(doc_id)
        assert meta is not None
        assert meta.filename == "uncataloged.txt"
        assert meta.total_chunks == 1

        # Clean up
        self.ingestion_service.vector_store.delete_document(doc_id)
        self.ingestion_service.registry.remove(doc_id)

    def test_repeated_deletion_idempotency_does_not_corrupt_stores(self, sync_client: TestClient):
        """Verify repeated deletion is idempotent and does not corrupt persistent stores or indexes."""
        content = b"Idempotent deletion test content."
        files = {"file": ("idempotent.txt", io.BytesIO(content), "text/plain")}
        res = sync_client.post(f"{self.settings.API_V1_STR}/documents/upload", files=files)
        assert res.status_code == 200
        doc_id = res.json()["document_id"]

        # First deletion -> 200 OK
        del1 = sync_client.delete(f"{self.settings.API_V1_STR}/documents/{doc_id}")
        assert del1.status_code == 200
        assert del1.json()["status"] == "deleted"

        # Second deletion -> 404 Not Found
        del2 = sync_client.delete(f"{self.settings.API_V1_STR}/documents/{doc_id}")
        assert del2.status_code == 404

        # Sequential reconciliation runs are idempotent
        r1 = self.ingestion_service.registry.reconcile_from_vector_store(self.ingestion_service.vector_store)
        r2 = self.ingestion_service.registry.reconcile_from_vector_store(self.ingestion_service.vector_store)
        assert r2 == 0
