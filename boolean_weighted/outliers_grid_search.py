import csv
import json
import itertools
import heapq
import os
import random

from outliers_scoring import calculate_score
from outliers_evaluation import evaluate_queries


FIELDS = ["title", "author", "bibliography", "body"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTLIERS_PROCESSED_DIR = os.path.join(
    BASE_DIR, "..", "outliers_preprocesing_feature_extraction", "outliers_preprocessed"
)
OUTLIERS_QRELS_PATH = os.path.join(
    BASE_DIR, "..", "outliers_preprocesing_feature_extraction", "outliers_raw", "cranqrel"
)


def load_csv(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_data(data_dir):
    # Read documents, queries, and qrels from the Outliers project. Qrels grade
    # -1 is unjudged and excluded; grades 1-4 are relevant for this evaluator.
    documents = load_csv(os.path.join(data_dir, "outliers_documents_processed.csv"))
    queries = load_csv(os.path.join(data_dir, "outliers_queries_processed.csv"))
    with open(OUTLIERS_QRELS_PATH, "r", encoding="utf-8") as qrels_file:
        qrels_rows = [line.split() for line in qrels_file if line.strip()]

    for document in documents:
        document["doc_id"] = int(document["doc_id"])

    for query in queries:
        query["query_id"] = int(query["query_id"])

    relevance = {}
    for row in qrels_rows:
        if len(row) != 3:
            raise ValueError(f"Invalid qrels row: {row!r}")
        query_id, doc_id, grade = map(int, row)
        if 1 <= grade <= 4:
            relevance.setdefault(query_id, set()).add(doc_id)

    query_map = {query["query_id"]: query for query in queries}
    # The Outliers folder has no saved split files, so create reproducible
    # query-level 70/15/15 splits directly from its processed query IDs.
    query_ids = sorted(query_map)
    shuffled_ids = query_ids.copy()
    random.Random(42).shuffle(shuffled_ids)
    val_count = round(len(shuffled_ids) * 0.15)
    test_count = round(len(shuffled_ids) * 0.15)
    train_query_ids = sorted(shuffled_ids[:len(shuffled_ids) - val_count - test_count])
    val_query_ids = sorted(shuffled_ids[len(train_query_ids):len(train_query_ids) + val_count])
    test_query_ids = sorted(shuffled_ids[len(train_query_ids) + val_count:])

    return (
        documents,
        query_map,
        relevance,
        train_query_ids,
        val_query_ids,
        test_query_ids,
    )


def token_set(tokens):
    """Convert CSV JSON tokens or token lists to a lowercase set."""
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    return {str(token).lower() for token in tokens}


def precompute_match_count_groups(documents, query_map, query_ids):
    """Precompute the number of query-term matches in each field once."""
    prepared_documents = []

    for document in documents:
        field_sets = [
            token_set(document[field + "_stemmed_tokens"])
            for field in FIELDS
        ]
        prepared_documents.append((document["doc_id"], field_sets))

    cached_counts = {}

    for query_id in query_ids:
        query_tokens = token_set(
            query_map[query_id]["query_stemmed_tokens"]
        )
        count_groups = {}

        for doc_id, field_sets in prepared_documents:
            match_counts = tuple(
                len(query_tokens & field_tokens)
                for field_tokens in field_sets
            )
            count_groups.setdefault(match_counts, []).append(doc_id)

        for doc_ids in count_groups.values():
            doc_ids.sort()

        cached_counts[query_id] = count_groups

    return cached_counts


def build_boolean_pattern_cache(cached_counts, min_matches):
    """Convert cached field match counts to Boolean 0/1 field patterns."""
    cached_patterns = {}

    for query_id, count_groups in cached_counts.items():
        boolean_groups = {}

        for match_counts, doc_ids in count_groups.items():
            pattern = tuple(
                int(count >= min_matches)
                for count in match_counts
            )
            # Keep exact match-count groups separate so that documents tied on
            # the Boolean score can be ordered by the strength of their match.
            boolean_groups.setdefault((pattern, match_counts), []).extend(doc_ids)

        for doc_ids in boolean_groups.values():
            doc_ids.sort()

        cached_patterns[query_id] = boolean_groups

    return cached_patterns


def rank_from_cached_patterns(cached_patterns, query_id, weights, k=None):
    """Rank by Boolean score, then weighted term matches, then document ID."""
    score_buckets = {}
    groups = cached_patterns[query_id]

    for (pattern, match_counts), doc_id_groups in groups.items():
        boolean_scores = {
            field: pattern[index]
            for index, field in enumerate(FIELDS)
        }
        score = calculate_score(boolean_scores, weights)
        match_strength = sum(
            weights[field] * match_counts[index]
            for index, field in enumerate(FIELDS)
        )

        if score > 0:
            score_buckets.setdefault(score, {}).setdefault(
                match_strength, []
            ).append(doc_id_groups)

    ranked = []

    for score in sorted(score_buckets, reverse=True):
        for match_strength in sorted(score_buckets[score], reverse=True):
            # Document ID is the final deterministic tie-break after comparing
            # both Boolean field score and weighted number of matched terms.
            id_groups = score_buckets[score][match_strength]
            for doc_id in heapq.merge(*id_groups):
                ranked.append({
                    "doc_id": doc_id,
                    "score": score,
                    "match_strength": match_strength,
                })
                if k is not None and len(ranked) >= k:
                    return ranked

    return ranked


def grid_search(cached_patterns, relevance, train_query_ids, weight_values, k=10):
    best_weights = None
    best_metrics = None
    best_f1 = -1.0

    total = len(weight_values) ** len(FIELDS) - 1
    checked = 0

    for combination in itertools.product(weight_values, repeat=len(FIELDS)):
        if not any(combination):
            continue

        checked += 1
        weights = {
            field: combination[index]
            for index, field in enumerate(FIELDS)
        }

        rankings = {
            query_id: rank_from_cached_patterns(
                cached_patterns,
                query_id,
                weights,
                k=k,
            )
            for query_id in train_query_ids
        }

        metrics = evaluate_queries(
            rankings,
            relevance,
            train_query_ids,
            k=k,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_weights = weights.copy()
            best_metrics = metrics.copy()

        if checked % 100 == 0:
            print(f"Checked {checked}/{total} weight combinations")

    return best_weights, best_metrics


def evaluate_split(cached_patterns, weights, relevance, query_ids, k=10):
    rankings = {
        query_id: rank_from_cached_patterns(
            cached_patterns,
            query_id,
            weights,
            k=k,
        )
        for query_id in query_ids
    }
    metrics = evaluate_queries(rankings, relevance, query_ids, k=k)
    return rankings, metrics


if __name__ == "__main__":
    DATA_DIR = OUTLIERS_PROCESSED_DIR
    K = 10
    WEIGHT_VALUES = [0, 1, 2, 3, 4, 5]
    THRESHOLDS = [1, 2, 3]

    (
        documents,
        query_map,
        relevance,
        train_query_ids,
        val_query_ids,
        test_query_ids,
    ) = load_data(DATA_DIR)

    print("Documents:", len(documents))
    print("Training queries:", len(train_query_ids))
    print("Validation queries:", len(val_query_ids))
    print("Test queries:", len(test_query_ids))

    all_query_ids = list(dict.fromkeys(
        train_query_ids + val_query_ids + test_query_ids
    ))

    best_configuration = None
    best_validation_f1 = -1.0
    threshold_results = []
    print("Precomputing query-term match counts once...")
    cached_counts = precompute_match_count_groups(
        documents,
        query_map,
        all_query_ids,
    )
    print("Match-count precomputation complete.")

    best_cache = None

    for threshold in THRESHOLDS:
        print("\n" + "=" * 55)
        print(f"BOOLEAN MATCH THRESHOLD: {threshold}")
        print("=" * 55)
        cached_patterns = build_boolean_pattern_cache(
            cached_counts,
            min_matches=threshold,
        )
        print("Learning field weights on training queries...")

        best_weights, train_metrics = grid_search(
            cached_patterns,
            relevance,
            train_query_ids,
            WEIGHT_VALUES,
            k=K,
        )

        _, val_metrics = evaluate_split(
            cached_patterns,
            best_weights,
            relevance,
            val_query_ids,
            k=K,
        )

        print("Best training weights:", best_weights)
        print("Training metrics:", train_metrics)
        print("Validation metrics:", val_metrics)

        threshold_results.append({
            "min_matches": threshold,
            "weights": best_weights.copy(),
            "train_metrics": train_metrics.copy(),
            "validation_metrics": val_metrics.copy(),
        })

        if val_metrics["f1"] > best_validation_f1:
            best_validation_f1 = val_metrics["f1"]
            best_configuration = {
                "min_matches": threshold,
                "weights": best_weights.copy(),
                "train_metrics": train_metrics.copy(),
                "validation_metrics": val_metrics.copy(),
            }
            best_cache = cached_patterns

    print("\n" + "=" * 55)
    print("SELECTED CONFIGURATION (based on validation F1)")
    print("=" * 55)
    print("Minimum matches per field:", best_configuration["min_matches"])
    print("Field weights:", best_configuration["weights"])
    print("Training metrics:", best_configuration["train_metrics"])
    print("Validation metrics:", best_configuration["validation_metrics"])

    # The test set is evaluated only after threshold and weights are selected.
    test_rankings, test_metrics = evaluate_split(
        best_cache,
        best_configuration["weights"],
        relevance,
        test_query_ids,
        k=K,
    )

    print("\nFinal test metrics:")
    print(test_metrics)

    # Persist the Boolean grid-search outcome alongside the separate soft-test
    # artifacts. This includes each threshold's best training result, the
    # validation-selected configuration, and its final test evaluation.
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "boolean_grid_search_results.json")
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(
            {
                "cutoff": K,
                "weight_values": WEIGHT_VALUES,
                "threshold_results": threshold_results,
                "selected_configuration": best_configuration,
                "test_metrics": test_metrics,
                "test_rankings": test_rankings,
            },
            results_file,
            indent=2,
        )
    print("Saved Boolean grid-search results to:", results_path)

    if test_query_ids:
        sample_query_id = test_query_ids[0]
        sample_query = json.loads(
            query_map[sample_query_id]["query_stemmed_tokens"]
        )
        relevant_docs = relevance.get(sample_query_id, set())

        print("\nSample ranked results")
        print("Query ID:", sample_query_id)
        print("Query tokens:", sample_query)
        print("Top", K, "documents:")

        for result in test_rankings[sample_query_id]:
            status = "RELEVANT" if result["doc_id"] in relevant_docs else "not judged relevant"
            print(
                f"Document {result['doc_id']} | "
                f"Score {result['score']} | {status}"
            )
