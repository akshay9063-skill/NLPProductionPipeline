
import json

import numpy as np
import pytest

import ingest.search_documents as search_module


def test_load_embeddings_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        search_module,
        "EMBEDDINGS_FILE",
        tmp_path / "missing.jsonl",
    )

    with pytest.raises(FileNotFoundError):
        search_module.load_embeddings()


def test_load_embeddings_rejects_empty_file(tmp_path, monkeypatch):
    file_path = tmp_path / "embeddings.jsonl"
    file_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(search_module, "EMBEDDINGS_FILE", file_path)

    with pytest.raises(ValueError, match="No embeddings"):
        search_module.load_embeddings()


def test_search_returns_empty_for_nonpositive_top_k():
    assert search_module.search_documents("password reset", top_k=0) == []
    assert search_module.search_documents("password reset", top_k=-1) == []


def test_search_returns_empty_when_no_records_match_role(monkeypatch):
    records = [
        {
            "chunk_id": "PRIVATE_chunk_0000",
            "document_id": "PRIVATE",
            "title": "Private",
            "source": "private.md",
            "access_level": "restricted",
            "text": "Private information.",
            "embedding": [1.0, 0.0],
        }
    ]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        search_module, "load_embeddings", lambda: (records, vectors)
    )

    result = search_module.search_documents(
        "private information",
        allowed_access_levels={"public"},
    )

    assert result == []


def test_search_returns_ranked_results_and_respects_access(monkeypatch):
    records = [
        {
            "chunk_id": "DOC001_chunk_0000",
            "document_id": "DOC001",
            "title": "Password Help",
            "source": "password.md",
            "access_level": "public",
            "text": "Use Forgot Password to reset your password.",
            "embedding": [1.0, 0.0],
        },
        {
            "chunk_id": "DOC002_chunk_0000",
            "document_id": "DOC002",
            "title": "Restricted Help",
            "source": "restricted.md",
            "access_level": "restricted",
            "text": "Restricted password instructions.",
            "embedding": [0.0, 1.0],
        },
    ]
    vectors = np.array(
        [record["embedding"] for record in records],
        dtype=np.float32,
    )

    class FakeModel:
        def encode(self, questions, normalize_embeddings=True):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        search_module, "load_embeddings", lambda: (records, vectors)
    )
    monkeypatch.setattr(
        search_module, "SentenceTransformer", lambda name: FakeModel()
    )

    results = search_module.search_documents(
        "How do I reset my password?",
        top_k=3,
        allowed_access_levels={"public"},
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "DOC001"
    assert results[0]["access_level"] == "public"
    assert results[0]["score"] == pytest.approx(1.0)


def test_build_index_saves_faiss_index(tmp_path, monkeypatch):
    records = [
        {
            "chunk_id": "DOC001_chunk_0000",
            "document_id": "DOC001",
            "title": "Help",
            "source": "help.md",
            "access_level": "public",
            "text": "Helpful content.",
            "embedding": [1.0, 0.0],
        }
    ]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    index_path = tmp_path / "documents.faiss"

    monkeypatch.setattr(
        search_module, "load_embeddings", lambda: (records, vectors)
    )
    monkeypatch.setattr(search_module, "INDEX_FILE", index_path)

    returned_records, index = search_module.build_index()

    assert returned_records == records
    assert index.ntotal == 1
    assert index_path.exists()
