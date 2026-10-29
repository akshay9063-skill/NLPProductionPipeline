
from fastapi.testclient import TestClient

from backend.main import app
import backend.qa as qa_module


PUBLIC_HEADERS = {
    "X-API-Key": "local-public-key-change-this"
}

client = TestClient(app)


def test_qa_returns_answer_with_valid_citation(monkeypatch):
    def fake_search(question, top_k=3, **kwargs):
        return [
            {
                "document_id": "DOC001",
                "chunk_id": "DOC001_chunk_0000",
                "title": "Getting Started",
                "source": "data/documents/getting_started.md",
                "access_level": "public",
                "text": (
                    "Select Forgot Password on the login page. "
                    "A reset link will be sent to your email."
                ),
                "score": 0.70,
            }
        ]

    monkeypatch.setattr(
        qa_module, "search_documents", fake_search
    )

    response = client.post(
        "/qa",
        json={"question": "How do I reset my password?"},
        headers=PUBLIC_HEADERS,
    )

    assert response.status_code == 200

    data = response.json()
    assert data["refused"] is False
    assert "Forgot Password" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["document_id"] == "DOC001"
    assert data["citations"][0]["chunk_id"] == "DOC001_chunk_0000"


def test_qa_refuses_when_no_documents_are_retrieved(monkeypatch):
    monkeypatch.setattr(
        qa_module,
        "search_documents",
        lambda question, top_k=3, **kwargs: [],
    )

    response = client.post(
        "/qa",
        json={"question": "How does an unknown feature work?"},
        headers=PUBLIC_HEADERS,
    )

    assert response.status_code == 200

    data = response.json()
    assert data["refused"] is True
    assert data["citations"] == []


def test_qa_refuses_when_similarity_is_too_low(monkeypatch):
    def fake_search(question, top_k=3, **kwargs):
        return [
            {
                "document_id": "DOC002",
                "chunk_id": "DOC002_chunk_0000",
                "title": "Troubleshooting",
                "source": "data/documents/troubleshooting.md",
                "access_level": "public",
                "text": "Check your network connection.",
                "score": 0.10,
            }
        ]

    monkeypatch.setattr(
        qa_module, "search_documents", fake_search
    )

    response = client.post(
        "/qa",
        json={"question": "Explain an unrelated topic to me."},
        headers=PUBLIC_HEADERS,
    )

    assert response.status_code == 200

    data = response.json()
    assert data["refused"] is True
    assert data["citations"] == []


def test_qa_accepts_a_relevant_result_at_threshold(monkeypatch):
    def fake_search(question, top_k=3, **kwargs):
        return [
            {
                "document_id": "DOC001",
                "chunk_id": "DOC001_chunk_0000",
                "title": "Getting Started",
                "source": "data/documents/getting_started.md",
                "access_level": "public",
                "text": "Use Forgot Password to reset your password.",
                "score": qa_module.MIN_SIMILARITY,
            }
        ]

    monkeypatch.setattr(
        qa_module, "search_documents", fake_search
    )

    response = client.post(
        "/qa",
        json={"question": "How can I reset my password?"},
        headers=PUBLIC_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["refused"] is False
