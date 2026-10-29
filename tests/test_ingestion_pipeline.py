
import json

import numpy as np
import pytest

import ingest.load_documents as loader
import ingest.chunk_documents as chunker
import ingest.embed_documents as embedder


def test_parse_document_with_metadata_and_title(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text(
        "---\ndocument_id: DOC100\ntitle: My Guide\n"
        "access_level: public\n---\n# Heading\nUseful content.",
        encoding="utf-8",
    )

    result = loader.parse_document(path)

    assert result["document_id"] == "DOC100"
    assert result["title"] == "My Guide"
    assert result["access_level"] == "public"
    assert result["content"] == "# Heading\nUseful content."
    assert result["source"] == "guide.md"


def test_parse_document_uses_defaults(tmp_path):
    path = tmp_path / "simple_guide.txt"
    path.write_text("Plain text content.", encoding="utf-8")

    result = loader.parse_document(path)

    assert result["document_id"] == "simple_guide"
    assert result["title"] == "Simple Guide"
    assert result["access_level"] == "restricted"
    assert result["content"] == "Plain text content."


def test_parse_document_uses_heading_as_title(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Heading Title\nSome content.", encoding="utf-8")

    result = loader.parse_document(path)

    assert result["title"] == "Heading Title"


def test_load_documents_skips_empty_document(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "empty.md").write_text("   \n", encoding="utf-8")
    (docs / "valid.md").write_text("Useful content.", encoding="utf-8")

    output = tmp_path / "processed" / "documents.jsonl"
    monkeypatch.setattr(loader, "DOCUMENTS_DIR", docs)
    monkeypatch.setattr(loader, "OUTPUT_FILE", output)

    assert loader.load_documents() == 1

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["document_id"] == "valid"


def test_load_documents_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        loader, "DOCUMENTS_DIR", tmp_path / "missing"
    )

    with pytest.raises(FileNotFoundError):
        loader.load_documents()


def test_split_text_empty_and_short_text():
    assert chunker.split_text("") == []
    assert chunker.split_text("  ") == []
    assert chunker.split_text("short text") == ["short text"]


def test_split_text_creates_overlapping_chunks():
    text = "word " * 300

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= chunker.CHUNK_SIZE for chunk in chunks)


def test_chunk_documents_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(
        chunker, "INPUT_FILE", tmp_path / "missing.jsonl"
    )

    with pytest.raises(FileNotFoundError):
        chunker.chunk_documents()


def test_chunk_documents_writes_chunk_metadata(tmp_path, monkeypatch):
    input_file = tmp_path / "documents.jsonl"
    output_file = tmp_path / "chunks.jsonl"

    document = {
        "document_id": "DOC100",
        "title": "Guide",
        "source": "guide.md",
        "access_level": "public",
        "content": "A useful guide.",
    }
    input_file.write_text(json.dumps(document) + "\n", encoding="utf-8")

    monkeypatch.setattr(chunker, "INPUT_FILE", input_file)
    monkeypatch.setattr(chunker, "OUTPUT_FILE", output_file)

    assert chunker.chunk_documents() == 1

    record = json.loads(output_file.read_text(encoding="utf-8").strip())
    assert record["chunk_id"] == "DOC100_chunk_0000"
    assert record["document_id"] == "DOC100"
    assert record["access_level"] == "public"
    assert record["text"] == "A useful guide."


def test_embed_documents_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(
        embedder, "INPUT_FILE", tmp_path / "missing.jsonl"
    )

    with pytest.raises(FileNotFoundError):
        embedder.embed_documents()


def test_embed_documents_rejects_empty_input(tmp_path, monkeypatch):
    input_file = tmp_path / "chunks.jsonl"
    input_file.write_text("\n  \n", encoding="utf-8")

    monkeypatch.setattr(embedder, "INPUT_FILE", input_file)

    with pytest.raises(ValueError, match="No document chunks"):
        embedder.embed_documents()


def test_embed_documents_writes_mock_embeddings(tmp_path, monkeypatch):
    input_file = tmp_path / "chunks.jsonl"
    output_file = tmp_path / "embeddings.jsonl"

    records = [
        {
            "chunk_id": "DOC100_chunk_0000",
            "document_id": "DOC100",
            "title": "Guide",
            "source": "guide.md",
            "access_level": "public",
            "text": "Useful content.",
        }
    ]
    input_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    class FakeModel:
        def encode(self, texts, **kwargs):
            assert texts == ["Useful content."]
            return np.array([[0.6, 0.8]], dtype=np.float32)

    monkeypatch.setattr(embedder, "INPUT_FILE", input_file)
    monkeypatch.setattr(embedder, "OUTPUT_FILE", output_file)
    monkeypatch.setattr(embedder, "SentenceTransformer", lambda name: FakeModel())

    assert embedder.embed_documents() == 1

    result = json.loads(output_file.read_text(encoding="utf-8").strip())
    assert result["document_id"] == "DOC100"
    assert result["embedding"] == pytest.approx([0.6, 0.8])
