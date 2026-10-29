
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"


def parse_document(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig").lstrip()
    metadata = {}

    # Extract the metadata header and remove it from the content.
    if text.startswith("---"):
        parts = text.split("---", 2)

        if len(parts) == 3 and not parts[0].strip():
            metadata_text = parts[1]

            for line in metadata_text.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

            text = parts[2].strip()

    title_match = re.search(r"(?m)^#\s+(.+)$", text)

    title = metadata.get(
        "title",
        title_match.group(1).strip()
        if title_match
        else path.stem.replace("_", " ").title(),
    )

    # Use a project-relative path for project documents.
    # Temporary test files use their filename.
    source = (
        str(path.relative_to(PROJECT_ROOT))
        if path.is_relative_to(PROJECT_ROOT)
        else path.name
    )

    return {
        "document_id": metadata.get("document_id", path.stem),
        "title": title,
        "source": source,
        "access_level": metadata.get("access_level", "restricted"),
        "content": text.strip(),
    }


def load_documents() -> int:
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documentation directory not found: {DOCUMENTS_DIR}"
        )

    files = sorted(
        list(DOCUMENTS_DIR.rglob("*.md"))
        + list(DOCUMENTS_DIR.rglob("*.txt"))
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for path in files:
            document = parse_document(path)

            if not document["content"]:
                print(f"Skipping empty document: {path.name}")
                continue

            output.write(
                json.dumps(document, ensure_ascii=False) + "\n"
            )
            count += 1

    print(f"Documents loaded: {count}")
    print(f"Output written to: {OUTPUT_FILE}")

    return count


if __name__ == "__main__":
    load_documents()
