
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import app
from backend.security import get_current_role

client = TestClient(app)


def test_missing_api_key_configuration_returns_503(monkeypatch):
    monkeypatch.setenv("PUBLIC_API_KEY", "")
    monkeypatch.setenv("STAFF_API_KEY", "")

    response = client.post(
        "/qa",
        json={"question": "How do I reset my password?"},
        headers={"X-API-Key": "any-key"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "API authentication is not configured."
    )


def test_staff_key_is_recognized(monkeypatch):
    monkeypatch.setenv("PUBLIC_API_KEY", "test-public-key")
    monkeypatch.setenv("STAFF_API_KEY", "test-staff-key")

    assert get_current_role("test-staff-key") == "staff"


def test_public_key_is_recognized(monkeypatch):
    monkeypatch.setenv("PUBLIC_API_KEY", "test-public-key")
    monkeypatch.setenv("STAFF_API_KEY", "test-staff-key")

    assert get_current_role("test-public-key") == "public"


def test_invalid_key_raises_401(monkeypatch):
    monkeypatch.setenv("PUBLIC_API_KEY", "test-public-key")
    monkeypatch.setenv("STAFF_API_KEY", "test-staff-key")

    with pytest.raises(HTTPException) as error:
        get_current_role("invalid-key")

    assert error.value.status_code == 401
