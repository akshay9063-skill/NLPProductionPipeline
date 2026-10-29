
import json
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from ingest.search_documents import search_documents

client = TestClient(app)


def test_qa_rejects_missing_api_key():
    response = client.post(
        "/qa",
        json={"question": "How do I reset my password?"},
    )
    assert response.status_code == 401


def test_qa_rejects_invalid_api_key():
    response = client.post(
        "/qa",
        json={"question": "How do I reset my password?"},
        headers={"X-API-Key": "invalid-key"},
    )
    assert response.status_code == 401


def test_public_search_excludes_restricted_documents():
    results = search_documents(
        "How do I reset my password?",
        top_k=10,
        allowed_access_levels={"public"},
    )
    assert all(
        result["access_level"] == "public"
        for result in results
    )


def test_restricted_document_is_not_returned_to_public_search(
    tmp_path, monkeypatch
):
    import ingest.search_documents as search_module

    record = {
        "document_id": "PRIVATE001",
        "chunk_id": "PRIVATE001_chunk_0000",
        "title": "Private Test Document",
        "source": "synthetic_private_test.md",
        "access_level": "restricted",
        "text": "SYNTHETIC_PRIVATE_MARKER_83921",
        "embedding": [1.0, 0.0],
    }

    embeddings_file = tmp_path / "embeddings.jsonl"
    embeddings_file.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        search_module, "EMBEDDINGS_FILE", embeddings_file
    )

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        search_module, "SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    public_results = search_documents(
        "private test document",
        allowed_access_levels={"public"},
    )
    assert public_results == []

    staff_results = search_documents(
        "private test document",
        allowed_access_levels={"restricted"},
    )
    assert len(staff_results) == 1
    assert staff_results[0]["document_id"] == "PRIVATE001"
