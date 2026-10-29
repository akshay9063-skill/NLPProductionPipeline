
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "embeddings.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_documents() -> int:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Chunks file not found. Run the loader and chunker first."
        )

    records = []

    with INPUT_FILE.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("No document chunks were found to embed.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [record["text"] for record in records]

    print(f"Generating embeddings for {len(texts)} chunks...")
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record, vector in zip(records, vectors):
            output_record = {
                **record,
                "embedding": vector.tolist(),
            }
            file.write(
                json.dumps(output_record, ensure_ascii=False) + "\n"
            )

    print(f"Embeddings created: {len(records)}")
    print(f"Output written to: {OUTPUT_FILE}")

    return len(records)


if __name__ == "__main__":
    embed_documents()
