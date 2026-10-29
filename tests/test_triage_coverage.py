
from backend.triage import triage_ticket


def test_security_ticket_is_critical():
    result = triage_ticket(
        "We discovered a security breach affecting customer records."
    )

    assert result["category"] == "security"
    assert result["priority"] == "critical"
    assert result["product_area"] == "security"


def test_bug_ticket_is_high_priority():
    result = triage_ticket(
        "The login page crashes when I submit the form."
    )

    assert result["category"] == "bug"
    assert result["priority"] == "high"
    assert result["product_area"] == "application"


def test_performance_ticket_is_medium_priority():
    result = triage_ticket(
        "The application is slow and has high latency."
    )

    assert result["category"] == "performance"
    assert result["priority"] == "medium"
    assert result["product_area"] == "performance"


def test_unclassified_ticket_uses_default_category():
    result = triage_ticket(
        "Please add a new theme to the settings page."
    )

    assert result["category"] == "other"
    assert result["priority"] == "low"
    assert result["product_area"] == "general"


def test_ticket_summary_is_limited_to_200_characters():
    ticket = "A" * 250

    result = triage_ticket(ticket)

    assert len(result["summary"]) == 200
