
from backend.qa import answer_question


def test_unknown_role_is_refused():
    result = answer_question(
        "How do I reset my password?",
        role="unknown",
    )

    assert result["refused"] is True
    assert result["citations"] == []
    assert result["answer"] == "Access denied."


def test_no_search_results_are_refused(monkeypatch):
    import backend.qa as qa_module

    monkeypatch.setattr(
        qa_module,
        "search_documents",
        lambda *args, **kwargs: [],
    )

    result = answer_question("How do I reset my password?")

    assert result["refused"] is True
    assert result["citations"] == []


def test_low_similarity_results_are_refused(monkeypatch):
    import backend.qa as qa_module

    monkeypatch.setattr(
        qa_module,
        "search_documents",
        lambda *args, **kwargs: [
            {
                "score": 0.01,
                "title": "Unrelated document",
                "text": "An unrelated passage.",
                "document_id": "DOC999",
                "chunk_id": "DOC999_chunk_0000",
            }
        ],
    )

    result = answer_question("How do I reset my password?")

    assert result["refused"] is True
    assert result["citations"] == []
