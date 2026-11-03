from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

PREPROCESSING_DIR = ROOT / "outliers_SVM_MethodB" / "outliers_preprocesing_feature_extraction"

PROCESSED_DIR = PREPROCESSING_DIR / "outliers_preprocessed"
RAW_DIR = PREPROCESSING_DIR / "outliers_raw"
STOPWORDS_PATH = RAW_DIR / "stopwords.txt"

MODEL_DIR = ROOT / "outliers_SVM_MethodB" / "outliers_svm_models_tuned"
MODEL_PATH = MODEL_DIR / "outliers_ranking_svm_tuned.joblib"
METADATA_PATH = MODEL_DIR / "outliers_tuned_model_metadata.json"

OUTPUT_PATH = ROOT / "svm_query_results.txt"

DEFAULT_TOP_K = 10

# Import the existing preprocessing modules.
sys.path.insert(0, str(PREPROCESSING_DIR))

from outliers_feature_extractor import FEATURE_NAMES, FeatureExtractor
from outliers_porter import PorterStemmer
from outliers_preprocess import preprocess_text, load_stemmed_stopwords


def load_system():
    print("Loading trained Ranking SVM...")
    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_columns = metadata["feature_columns"]
    if feature_columns != list(FEATURE_NAMES):
        raise ValueError("Feature order mismatch.")

    print("Loading Cranfield documents...")

    extractor = FeatureExtractor.from_processed(PROCESSED_DIR)
    stemmer = PorterStemmer()
    stopwords = load_stemmed_stopwords(STOPWORDS_PATH, stemmer)

    print("System loaded successfully.")
    print("Documents:", extractor.num_documents)
    print("Features:", len(feature_columns))
    print("Selected C:", metadata["best_C"])

    return model, extractor, stemmer, stopwords, feature_columns


def process_query(query, stemmer, stopwords):
    return preprocess_text(query, stemmer, stopwords)


def rank_documents(
    query_tokens,
    model,
    extractor,
    feature_columns
):

    feature_rows = []
    document_ids = []

    for doc_id in extractor.document_ids:

        features = extractor.extract(query_tokens, doc_id)
        feature_rows.append([
            features[name]
            for name in feature_columns
        ])

        document_ids.append(doc_id)

    X = np.asarray(feature_rows, dtype=np.float64)

    # Saved pipeline applies StandardScaler and LinearSVC.
    scores = model.decision_function(X)
    results = pd.DataFrame({
        "document_id": document_ids,
        "score": scores
    })

    results = results.sort_values(
        "score",
        ascending=False,
        kind="stable"
    ).reset_index(drop=True)

    results.insert(
        0,
        "rank",
        np.arange(1, len(results) + 1)
    )

    return results


def save_results(results, output_path):

    with Path(output_path).open("w", encoding="utf-8") as f:
        for doc_id in results["document_id"]:
            f.write(f"{int(doc_id)}\n")

    print("\nResults saved to:", output_path)


def main():
    parser = argparse.ArgumentParser(description="Cranfield Retrieval using Ranking SVM")
    parser.add_argument("--query",type=str,help="Search query")
    parser.add_argument("--query-file",type=Path,help="File containing one query per line")
    parser.add_argument("--top-k",type=int,default=DEFAULT_TOP_K,help="Number of top documents to output per query")
    parser.add_argument("--output",type=Path,default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive.")

    if args.query and args.query_file:
        parser.error("Use either --query or --query-file.")

    model, extractor, stemmer, stopwords, feature_columns = load_system()

    if not args.query_file:
        query = args.query
        if query is None:
            query = input("\nEnter search query: ")
        query = query.strip()
        if not query:
            parser.error("Query cannot be empty.")

        query_tokens = process_query(
            query,
            stemmer,
            stopwords
        )

        if not query_tokens:
            parser.error("No tokens remain after preprocessing.")

        results = rank_documents(
            query_tokens,
            model,
            extractor,
            feature_columns
        )

        print("\nQuery:", query)
        print("Processed query:", query_tokens)

        display = (
            results.head(args.top_k)
            if args.top_k is not None
            else results
        )

        print("\nRanked documents:")
        print(display.to_string(index=False))

        # Save only the top K documents.
        save_results(display, args.output)

    else:
        if not args.query_file.exists():
            parser.error("Query file does not exist.")

        with args.query_file.open("r", encoding="utf-8") as f:
            queries = [
                line.strip()
                for line in f
                if line.strip()
            ]

        with args.output.open("w", encoding="utf-8") as output:
            for qnum, query in enumerate(queries, start=1):
                query_tokens = process_query(
                    query,
                    stemmer,
                    stopwords
                )

                if not query_tokens:
                    print(f"Skipping query {qnum}: no tokens.")
                    continue

                results = rank_documents(
                    query_tokens,
                    model,
                    extractor,
                    feature_columns
                )

                if args.top_k is not None:
                    results = results.head(args.top_k)

                output.write(f"Query {qnum}\n")

                for doc_id in results["document_id"]:
                    output.write(f"{int(doc_id)}\n")

                output.write("\n")
                print(f"Completed query {qnum}/{len(queries)}")
        print("\nBatch results saved to:", args.output)


if __name__ == "__main__":
    main()
