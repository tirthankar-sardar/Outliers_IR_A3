"""Interactively try free-text queries with the tuned Boolean weighted model.

Run from any directory with ``python3 boolean_weighted/outleirs_testA.py``.
Queries are normalized, Porter-stemmed, and stripped of the collection's
stopwords before being matched against the processed Cranfield documents.
"""

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PREPROCESSING_DIR = PROJECT_DIR / "outliers_preprocesing_feature_extraction"
PROCESSED_DIR = PREPROCESSING_DIR / "outliers_preprocessed"
DOCUMENTS_PATH = PROCESSED_DIR / "outliers_documents_processed.csv"
STOPWORDS_PATH = PREPROCESSING_DIR / "outliers_raw" / "stopwords.txt"

# Best configuration saved by boolean_weighted/grid_search.py.
WEIGHTS = {"title": 1, "author": 0, "bibliography": 0, "body": 0}
MIN_MATCHES = 3
DEFAULT_LIMIT = 10

# Reuse the exact tokenizer, normalization, stemming, and stopword logic used
# when the document collection was preprocessed.
sys.path.insert(0, str(PREPROCESSING_DIR))
from outliers_porter import PorterStemmer  # noqa: E402
from outliers_preprocess import (  # noqa: E402
    load_stemmed_stopwords,
    normalize,
    stem_tokens,
    tokenize,
)

from boolean_matching import get_boolean_scores  # noqa: E402
from scoring import calculate_score  # noqa: E402


def load_documents(path=DOCUMENTS_PATH):
    """Read the preprocessed collection and decode its field tokens."""
    documents = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            row["doc_id"] = int(row["doc_id"])
            for field in ("title", "author", "bibliography", "body"):
                row[field + "_stemmed_tokens"] = json.loads(
                    row[field + "_stemmed_tokens"]
                )
            documents.append(row)
    return documents


def prepare_query(query, stemmer, stopwords):
    tokens = normalize(tokenize(query))
    tokens = stem_tokens(tokens, stemmer)
    return [token for token in tokens if token not in stopwords]


def rank_query(query_tokens, documents, limit=DEFAULT_LIMIT):
    """Rank matching documents with the tuned Boolean field weights."""
    matches = []
    for document in documents:
        field_scores = get_boolean_scores(
            query_tokens, document, min_matches=MIN_MATCHES
        )
        score = calculate_score(field_scores, WEIGHTS)
        if score > 0:
            matches.append((score, document))
    matches.sort(key=lambda result: (-result[0], result[1]["doc_id"]))
    return matches[:limit]


def main():
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Processed documents not found: {DOCUMENTS_PATH}\n"
            "Run outliers_preprocesing_feature_extraction/outliers_preprocess.py first."
        )

    stemmer = PorterStemmer()
    stopwords = load_stemmed_stopwords(STOPWORDS_PATH, stemmer)
    documents = load_documents()

    print("Boolean weighted query tester")
    print(f"Model: min {MIN_MATCHES} distinct query-term matches in the title")
    print("Enter a query, or type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        query_tokens = prepare_query(query, stemmer, stopwords)
        if not query_tokens:
            print("No searchable terms remain after preprocessing.\n")
            continue

        results = rank_query(query_tokens, documents)
        print("Processed terms:", ", ".join(query_tokens))
        if not results:
            print("No documents matched.\n")
            continue

        for rank, (score, document) in enumerate(results, start=1):
            title = document.get("title_clean", "").strip()
            body = " ".join(document.get("body_clean", "").split())
            snippet = body[:220] + ("..." if len(body) > 220 else "")
            print(f"{rank}. Document {document['doc_id']} (score: {score})")
            print(f"   Title: {title or '(no title)'}")
            print(f"   {snippet}")
        print()


if __name__ == "__main__":
    main()
