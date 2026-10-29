
import re

from ingest.search_documents import search_documents


MIN_SIMILARITY = 0.25

ROLE_ACCESS = {
    "public": {"public"},
    "staff": {"public", "restricted"},
}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can",
    "do", "does", "for", "from", "how", "i", "if", "in",
    "is", "it", "me", "my", "of", "on", "or", "should",
    "the", "to", "what", "when", "where", "which", "who",
    "why", "with",
}

SYNONYMS = {
    "reset": {"reset", "forgot", "password"},
    "forgot": {"forgot", "reset", "password"},
    "login": {"login", "sign", "account", "fails", "failure"},
    "fail": {"fail", "fails", "failed", "failure", "error"},
    "fails": {"fail", "fails", "failed", "failure", "error"},
    "failed": {"fail", "fails", "failed", "failure", "error"},
    "slow": {"slow", "slowly", "performance", "latency"},
    "performance": {"performance", "slow", "slowly", "latency"},
    "upload": {"upload", "file"},
    "account": {"account", "registration", "create"},
    "create": {"create", "creating", "registration", "account"},
    "report": {"report", "contact", "support"},
}


def _tokens(text: str) -> set[str]:
    """Return meaningful lowercase tokens from text."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS and len(word) > 1
    }


def _expand_question_tokens(question: str) -> set[str]:
    """Expand question terms using a small synonym dictionary."""
    tokens = _tokens(question)
    expanded = set(tokens)

    for token in tokens:
        expanded.update(SYNONYMS.get(token, set()))

    return expanded


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown content into heading/body sections."""
    sections = []
    current_heading = ""
    current_lines = []

    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)

        if match:
            body = "\n".join(current_lines).strip()

            if current_heading or body:
                sections.append((current_heading, body))

            current_heading = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()

    if current_heading or body:
        sections.append((current_heading, body))

    return sections


def _clean_markdown(text: str) -> str:
    """Remove Markdown headings and horizontal separators."""
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into nonempty sentences."""
    cleaned = _clean_markdown(text)

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        if sentence.strip()
    ]


def _section_score(
    question_tokens: set[str],
    heading: str,
    body: str,
) -> float:
    """Score a section using its heading and body."""
    if not question_tokens:
        return 0.0

    heading_overlap = question_tokens & _tokens(heading)
    body_overlap = question_tokens & _tokens(body)

    heading_score = len(heading_overlap) / len(question_tokens)
    body_score = len(body_overlap) / len(question_tokens)

    return 2.0 * heading_score + body_score


def _extract_relevant_sentences(question: str, text: str) -> str:
    """
    Select a relevant section and return useful sentences from it.

    If no question terms match the text, retain the legacy behavior of
    returning only the first sentence.
    """
    sentences = _split_sentences(text)

    if not sentences:
        return text.strip()

    question_tokens = _expand_question_tokens(question)

    if not question_tokens:
        return sentences[0]

    sections = _split_sections(text)

    # Preserve the original fallback expected by the existing tests.
    all_tokens = _tokens(_clean_markdown(text))

    if not (question_tokens & all_tokens):
        return sentences[0]

    if not sections:
        return sentences[0]

    scored_sections = []

    for index, (heading, body) in enumerate(sections):
        score = _section_score(question_tokens, heading, body)
        scored_sections.append((score, index))

    best_score, best_index = max(
        scored_sections,
        key=lambda item: item[0],
    )

    if best_score <= 0:
        return sentences[0]

    heading, body = sections[best_index]
    selected_text = _clean_markdown(f"{heading}\n{body}")
    selected_sentences = _split_sentences(selected_text)

    if not selected_sentences:
        return sentences[0]

    # Include the full matching section, including follow-up instructions.
    return " ".join(selected_sentences)


def answer_question(question: str, role: str = "public") -> dict:
    """Answer using documentation the caller is authorized to access."""
    allowed_levels = ROLE_ACCESS.get(role)

    if allowed_levels is None:
        return {
            "answer": "Access denied.",
            "citations": [],
            "refused": True,
        }

    results = search_documents(
        question,
        top_k=3,
        allowed_access_levels=allowed_levels,
    )

    if not results or results[0]["score"] < MIN_SIMILARITY:
        return {
            "answer": (
                "I couldn't find sufficiently relevant information in the "
                "documentation you are authorized to access."
            ),
            "citations": [],
            "refused": True,
        }

    best = results[0]

    focused_answer = _extract_relevant_sentences(
        question,
        best["text"],
    )

    return {
        "answer": f"According to {best['title']}: {focused_answer}",
        "citations": [
            {
                "document_id": best["document_id"],
                "chunk_id": best["chunk_id"],
            }
        ],
        "refused": False,
    }
