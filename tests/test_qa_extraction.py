
from backend.qa import _extract_relevant_sentences


def test_empty_text_returns_empty_answer():
    assert _extract_relevant_sentences("password reset", "") == ""


def test_question_without_useful_keywords_uses_first_sentence():
    result = _extract_relevant_sentences(
        "how do I do it",
        "First sentence explains the process. Second sentence adds details.",
    )

    assert result == "First sentence explains the process."


def test_no_matching_keywords_uses_first_sentence():
    result = _extract_relevant_sentences(
        "quantum physics",
        "Reset your password using the login page. Contact support if needed.",
    )

    assert result == "Reset your password using the login page."
