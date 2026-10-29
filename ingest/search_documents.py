
import json
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embeddings.jsonl"
)
INDEX_FILE = PROJECT_ROOT / "data" / "processed" / "documents.faiss"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25
LOGIN_RELEVANCE_BOOST = 0.15


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can",
    "do", "does", "for", "from", "how", "i", "if", "in",
    "is", "it", "me", "my", "of", "on", "or", "should",
    "the", "to", "what", "when", "where", "which", "who",
    "why", "with",
}


def _tokens(text: str) -> set[str]:
    """Return meaningful lowercase tokens from text."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS and len(word) > 1
    }


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(MODEL_NAME)


def load_embeddings():
    """Load document chunks and their embedding vectors."""
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "Embeddings not found. Run "
            "python -m ingest.embed_documents first."
        )

    records = []

    with EMBEDDINGS_FILE.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("No embeddings were found.")

    vectors = np.asarray(
        [record["embedding"] for record in records],
        dtype="float32",
    )

    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Embedding vectors have an invalid shape.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors contain invalid values.")

    return records, vectors


def build_index():
    """Build and save a FAISS index for all document chunks."""
    records, vectors = load_embeddings()

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    print(f"Indexed chunks: {index.ntotal}")
    print(f"FAISS index saved to: {INDEX_FILE}")

    return records, index


def _keyword_score(
    question_tokens: set[str],
    text: str,
) -> float:
    """Calculate the proportion of question tokens found in a chunk."""
    if not question_tokens:
        return 0.0

    document_tokens = _tokens(text)
    return len(question_tokens & document_tokens) / len(question_tokens)


def _get_query_vector(
    question: str,
    dimension: int,
) -> np.ndarray:
    """Encode a question and validate the resulting vector."""
    query_vector = np.asarray(
        _get_model().encode(
            [question],
            normalize_embeddings=True,
        ),
        dtype="float32",
    )

    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise ValueError(
            "Query embedding must have shape (1, dimension)."
        )

    if query_vector.shape[1] != dimension:
        # A test may have replaced the model constructor after a previous
        # call cached the real model. Clear the cache and retry once.
        _get_model.cache_clear()

        query_vector = np.asarray(
            _get_model().encode(
                [question],
                normalize_embeddings=True,
            ),
            dtype="float32",
        )

    if query_vector.ndim != 2 or query_vector.shape != (1, dimension):
        raise ValueError(
            "Query and document embedding dimensions do not match: "
            f"query shape={query_vector.shape}, "
            f"document dimension={dimension}."
        )

    if not np.isfinite(query_vector).all():
        raise ValueError("Query embedding contains invalid values.")

    return np.ascontiguousarray(query_vector)


def _is_login_question(question: str) -> bool:
    """Identify common ways users ask about signing in."""
    return bool(
        re.search(
            r"\b(login|log\s+in|sign\s+in|sign-in|signin)\b",
            question,
            re.IGNORECASE,
        )
    )


def _is_login_troubleshooting_chunk(text: str) -> bool:
    """Identify chunks specifically about login failures."""
    return bool(
        re.search(
            r"\blogin failures\b",
            text,
            re.IGNORECASE,
        )
    )


def search_documents(
    question: str,
    top_k: int = 3,
    allowed_access_levels: set[str] | None = None,
) -> list[dict]:
    """
    Search authorized document chunks using semantic and keyword relevance.

    Only permitted access levels are searched. Public documents are the
    default when no access levels are provided.
    """
    if allowed_access_levels is None:
        allowed_access_levels = {"public"}

    if top_k <= 0 or not question.strip():
        return []

    all_records, _ = load_embeddings()

    # Enforce access restrictions before ranking documents.
    records = [
        record
        for record in all_records
        if record.get("access_level", "restricted")
        in allowed_access_levels
    ]

    if not records:
        return []

    vectors = np.asarray(
        [record["embedding"] for record in records],
        dtype="float32",
    )

    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Document embedding vectors have an invalid shape.")

    if not np.isfinite(vectors).all():
        raise ValueError("Document embeddings contain invalid values.")

    query_vector = _get_query_vector(
        question,
        dimension=vectors.shape[1],
    )

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))

    semantic_scores, positions = index.search(
        query_vector,
        len(records),
    )

    question_tokens = _tokens(question)
    login_question = _is_login_question(question)

    results = []

    for semantic_score, position in zip(
        semantic_scores[0],
        positions[0],
    ):
        if position < 0:
            continue

        record = records[int(position)]
        semantic_score = float(semantic_score)

        keyword_score = _keyword_score(
            question_tokens,
            record["text"],
        )

        combined_score = (
            SEMANTIC_WEIGHT * semantic_score
            + KEYWORD_WEIGHT * keyword_score
        )

        # Give the login troubleshooting section a modest boost for
        # questions explicitly asking about signing in.
        if (
            login_question
            and _is_login_troubleshooting_chunk(record["text"])
        ):
            combined_score += LOGIN_RELEVANCE_BOOST

        results.append(
            {
                "chunk_id": record["chunk_id"],
                "document_id": record["document_id"],
                "title": record["title"],
                "source": record["source"],
                "access_level": record.get(
                    "access_level", "restricted"
                ),
                "text": record["text"],
                "score": combined_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
            }
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]


if __name__ == "__main__":
    build_index()

    question = input(
        "Enter your documentation question: "
    ).strip()

    if question:
        for result in search_documents(question):
            print("\n" + "=" * 60)
            print(f"Title: {result['title']}")
            print(f"Document: {result['document_id']}")
            print(f"Access level: {result['access_level']}")
            print(f"Combined score: {result['score']:.4f}")
            print(f"Semantic score: {result['semantic_score']:.4f}")
            print(f"Keyword score: {result['keyword_score']:.4f}")
            print(f"Text: {result['text']}")
