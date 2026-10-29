
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def split_text(text: str) -> list[str]:
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a word boundary.
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Move forward while retaining some overlap.
        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def chunk_documents() -> int:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Run 'python -m ingest.load_documents' first."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with INPUT_FILE.open(encoding="utf-8") as source, \
            OUTPUT_FILE.open("w", encoding="utf-8") as output:

        for line in source:
            document = json.loads(line)

            for index, chunk_text in enumerate(
                split_text(document["content"])
            ):
                chunk = {
                    "chunk_id": (
                        f"{document['document_id']}_chunk_{index:04d}"
                    ),
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "source": document["source"],
                    "access_level": document["access_level"],
                    "text": chunk_text,
                }

                output.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )
                count += 1

    print(f"Chunks created: {count}")
    print(f"Output written to: {OUTPUT_FILE}")

    return count


if __name__ == "__main__":
    chunk_documents()
