"""Interactively try free-text queries with the tuned Boolean weighted model.

Run from any directory with ``python3 outliers_testA.py``.
Queries are normalized, Porter-stemmed, and stripped of the collection's
stopwords before being matched against the processed Cranfield documents.
"""

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = BASE_DIR / "outliers_SVM_MethodB" / "outliers_preprocesing_feature_extraction"
PROCESSED_DIR = PREPROCESSING_DIR / "outliers_preprocessed"
DOCUMENTS_PATH = PROCESSED_DIR / "outliers_documents_processed.csv"
STOPWORDS_PATH = PREPROCESSING_DIR / "outliers_raw" / "stopwords.txt"
BOOLEAN_DIR = BASE_DIR / "boolean_weighted"
MODEL_RESULTS_PATH = BOOLEAN_DIR / "results" / "boolean_grid_search_results.json"
DEFAULT_LIMIT = 10

# Reuse the exact tokenizer, normalization, stemming, and stopword logic used
# when the document collection was preprocessed.
sys.path.insert(0, str(PREPROCESSING_DIR))
sys.path.insert(0, str(BOOLEAN_DIR))
from outliers_porter import PorterStemmer  # noqa: E402
from outliers_preprocess import (  # noqa: E402
    load_stemmed_stopwords,
    normalize,
    stem_tokens,
    tokenize,
)

from outliers_boolean_matching import get_boolean_scores  # noqa: E402
from outliers_scoring import calculate_score  # noqa: E402


def load_model(path=MODEL_RESULTS_PATH):
    """Load the validation-selected Boolean weights and match threshold."""
    with path.open("r", encoding="utf-8") as source:
        results = json.load(source)
    configuration = results["selected_configuration"]
    return configuration["weights"], int(configuration["min_matches"])


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


def rank_query(query_tokens, documents, weights, min_matches, limit=DEFAULT_LIMIT):
    """Rank matching documents with the tuned Boolean field weights."""
    matches = []
    for document in documents:
        field_scores = get_boolean_scores(
            query_tokens, document, min_matches=min_matches
        )
        score = calculate_score(field_scores, weights)
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
    weights, min_matches = load_model()

    print("Boolean weighted query tester")
    print(f"Model: min {min_matches} distinct query-term matches per field")
    print("Field weights:", weights)
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

        results = rank_query(query_tokens, documents, weights, min_matches)
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
