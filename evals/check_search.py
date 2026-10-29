
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingest.search_documents import search_documents

question = "What should I do if login fails?"

print(f"Question: {question}\n")

results = search_documents(
    question,
    top_k=3,
    allowed_access_levels={"public"},
)

for rank, result in enumerate(results, start=1):
    print("=" * 70)
    print(f"Rank: {rank}")
    print(f"Document: {result['document_id']}")
    print(f"Title: {result['title']}")
    print(f"Similarity score: {result['score']:.4f}")
    print("\nRetrieved text:")
    print(result["text"])
    print()
