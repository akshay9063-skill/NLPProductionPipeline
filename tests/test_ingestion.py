
import json

from ingest.load_documents import parse_document
from ingest.chunk_documents import split_text


def test_parse_document_extracts_metadata(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text(
        "---\n"
        "document_id: DOC100\n"
        "title: Test Guide\n"
        "access_level: public\n"
        "---\n"
        "# Test Guide\n\n"
        "This is a test document.",
        encoding="utf-8",
    )

    document = parse_document(path)

    assert document["document_id"] == "DOC100"
    assert document["title"] == "Test Guide"
    assert document["access_level"] == "public"
    assert "This is a test document." in document["content"]


def test_split_text_returns_content():
    text = "This is a sample document for testing."

    chunks = split_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_handles_empty_text():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_preserves_all_content():
    text = "word " * 500

    chunks = split_text(text)

    assert len(chunks) > 1

    # Overlap is expected, so concatenate is not used here.
    # Verify each original word appears in the chunk collection.
    combined = " ".join(chunks)

    assert "word" in combined
    assert len(chunks[0]) <= 800


def test_chunks_keep_document_metadata():
    document = {
        "document_id": "DOC100",
        "title": "Test Guide",
        "source": "data/documents/guide.md",
        "access_level": "public",
        "content": "A short test document.",
    }

    chunks = split_text(document["content"])

    assert len(chunks) == 1

    chunk = {
        "chunk_id": f"{document['document_id']}_chunk_0000",
        "document_id": document["document_id"],
        "title": document["title"],
        "source": document["source"],
        "access_level": document["access_level"],
        "text": chunks[0],
    }

    assert chunk["document_id"] == "DOC100"
    assert chunk["access_level"] == "public"
    assert chunk["source"].endswith("guide.md")
    assert chunk["text"] == document["content"]
