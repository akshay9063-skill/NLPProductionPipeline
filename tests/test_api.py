
from fastapi.testclient import TestClient

from backend.main import app


PUBLIC_HEADERS = {
    "X-API-Key": "local-public-key-change-this"
}

INVALID_HEADERS = {
    "X-API-Key": "invalid-key"
}

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "NLP Production Pipeline"
    assert data["version"] == "0.1.0"


def test_triage_bug_ticket():
    response = client.post(
        "/triage",
        json={
            "ticket": (
                "The application crashes when I upload a file."
            )
        },
        headers=PUBLIC_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["category"] == "bug"
    assert data["priority"] == "high"
    assert data["product_area"] == "application"
    assert "summary" in data


def test_triage_response_has_required_fields():
    response = client.post(
        "/triage",
        json={
            "ticket": "The application is running slowly."
        },
        headers=PUBLIC_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()

    required_fields = {
        "category",
        "priority",
        "product_area",
        "summary",
    }

    assert set(data.keys()) == required_fields


def test_triage_rejects_missing_api_key():
    response = client.post(
        "/triage",
        json={"ticket": "The application crashes when starting."},
    )

    assert response.status_code == 401


def test_triage_rejects_invalid_api_key():
    response = client.post(
        "/triage",
        json={"ticket": "The application crashes when starting."},
        headers=INVALID_HEADERS,
    )

    assert response.status_code == 401


def test_qa_endpoint_refuses_without_documentation():
    response = client.post(
        "/qa",
        json={
            "question": "How does the product work?"
        },
        headers=PUBLIC_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["refused"] is True
    assert data["citations"] == []
    assert data["answer"]


def test_invalid_short_question_is_rejected():
    response = client.post(
        "/qa",
        json={"question": "Hi"},
        headers=PUBLIC_HEADERS,
    )

    assert response.status_code == 422


def test_invalid_short_ticket_is_rejected():
    response = client.post(
        "/triage",
        json={"ticket": "Bug"},
        headers=PUBLIC_HEADERS,
    )

    assert response.status_code == 422
