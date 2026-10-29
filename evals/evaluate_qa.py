import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.qa import answer_question
from ingest.search_documents import search_documents

DATASET_FILE = PROJECT_ROOT / "evals" / "qa_dataset.json"


def evaluate():
    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8-sig"))

    retrieval_hits = 0
    citation_hits = 0
    keyword_hits = 0

    for case in dataset:
        question = case["question"]
        expected_doc = case["expected_document_id"]
        expected_keywords = case["expected_keywords"]

        results = search_documents(
            question,
            top_k=3,
            allowed_access_levels={"public"},
        )
        retrieved_ids = [item["document_id"] for item in results]
        retrieval_ok = expected_doc in retrieved_ids

        response = answer_question(question, role="public")
        citation_ids = [
            item["document_id"] for item in response["citations"]
        ]
        citation_ok = expected_doc in citation_ids

        answer_lower = response["answer"].lower()
        keywords_found = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in answer_lower
        ]
        keyword_ok = len(keywords_found) == len(expected_keywords)

        retrieval_hits += int(retrieval_ok)
        citation_hits += int(citation_ok)
        keyword_hits += int(keyword_ok)

        print(f"\nQuestion: {question}")
        print(f"Expected document: {expected_doc}")
        print(f"Retrieved document IDs: {retrieved_ids}")
        print(f"Retrieval success: {retrieval_ok}")
        print(f"Citation correct: {citation_ok}")
        print(f"Keywords found: {keywords_found}")
        print(f"Answer: {response['answer']}")

    total = len(dataset)

    if total == 0:
        raise ValueError("The evaluation dataset is empty.")

    print("\n" + "=" * 50)
    print(f"Evaluation cases: {total}")
    print(f"Retrieval success: {retrieval_hits}/{total} ({retrieval_hits / total:.0%})")
    print(f"Citation accuracy: {citation_hits}/{total} ({citation_hits / total:.0%})")
    print(f"Complete keyword coverage: {keyword_hits}/{total} ({keyword_hits / total:.0%})")


if __name__ == "__main__":
    evaluate()
