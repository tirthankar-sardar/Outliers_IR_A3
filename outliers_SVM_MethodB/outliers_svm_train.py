from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = (
    BASE_DIR
    / "outliers_preprocesing_feature_extraction"
    / "outliers_features"
    / "outliers_svm_training_dataset.csv"
)

MODEL_DIR = BASE_DIR / "outliers_svm_models"

MODEL_PATH = MODEL_DIR / "outliers_ranking_svm.joblib"
WEIGHTS_PATH = MODEL_DIR / "outliers_feature_weights.csv"
PREDICTIONS_PATH = MODEL_DIR / "outliers_test_predictions.csv"
EVALUATION_PATH = MODEL_DIR / "outliers_evaluation_results.csv"
METADATA_PATH = MODEL_DIR / "outliers_model_metadata.json"

QUERY_COLUMN = "qid"
DOCUMENT_COLUMN = "docid"
RELEVANCE_COLUMN = "relevance"

EXPECTED_FEATURE_COUNT = 22

RANDOM_STATE = 42
TEST_SIZE = 0.20
C_VALUE = 1.0
MAX_PAIRS_PER_QUERY = 2000
K = 10


def load_dataset(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    rename_map = {}

    if QUERY_COLUMN not in df.columns and "query_id" in df.columns:
        rename_map["query_id"] = QUERY_COLUMN

    if DOCUMENT_COLUMN not in df.columns and "doc_id" in df.columns:
        rename_map["doc_id"] = DOCUMENT_COLUMN

    df = df.rename(columns=rename_map)

    required = {
        QUERY_COLUMN,
        DOCUMENT_COLUMN,
        RELEVANCE_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    feature_columns = [
        col for col in df.columns
        if col not in required
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {len(feature_columns)}."
        )

    df[feature_columns] = df[feature_columns].apply(
        pd.to_numeric, errors="raise"
    )

    df[RELEVANCE_COLUMN] = pd.to_numeric(
        df[RELEVANCE_COLUMN], errors="raise"
    )

    if df[feature_columns].isnull().any().any():
        raise ValueError("Feature columns contain missing values.")

    if not np.isfinite(df[feature_columns].to_numpy()).all():
        raise ValueError("Feature columns contain infinite values.")

    if df[RELEVANCE_COLUMN].isnull().any():
        raise ValueError("Relevance contains missing values.")

    valid_labels = {0, 1, 2, 3, 4}

    actual_labels = set(
        df[RELEVANCE_COLUMN].unique()
    )

    if not actual_labels.issubset(valid_labels):
        raise ValueError(f"Unexpected relevance labels: {actual_labels}")

    if not np.equal(
        df[RELEVANCE_COLUMN].to_numpy(),
        np.floor(df[RELEVANCE_COLUMN].to_numpy())
    ).all():
        raise ValueError("Relevance labels must be integers.")

    if df.duplicated(
        subset=[QUERY_COLUMN, DOCUMENT_COLUMN]
    ).any():
        raise ValueError("Duplicate query-document pairs found.")

    df[RELEVANCE_COLUMN] = df[RELEVANCE_COLUMN].astype(int)

    df[QUERY_COLUMN] = df[QUERY_COLUMN].astype(str)
    df[DOCUMENT_COLUMN] = df[DOCUMENT_COLUMN].astype(str)

    print(f"Dataset rows: {len(df)}")
    print(f"Queries: {df[QUERY_COLUMN].nunique()}")
    print(f"Documents per query: {df.groupby(QUERY_COLUMN).size().mean():.1f}")
    print(f"Features: {len(feature_columns)}")
    print(f"Relevance labels: {sorted(actual_labels)}")

    print("\nRelevance distribution:")
    print(df[RELEVANCE_COLUMN].value_counts().sort_index())

    return df, feature_columns


def split_by_query(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(
            df,
            groups=df[QUERY_COLUMN]
        )
    )

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    train_queries = set(train_df[QUERY_COLUMN])
    test_queries = set(test_df[QUERY_COLUMN])

    if not train_queries.isdisjoint(test_queries):
        raise ValueError("Query leakage detected.")

    print("\nQuery-level split:")
    print(f"Training queries: {len(train_queries)}")
    print(f"Testing queries: {len(test_queries)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")

    return train_df, test_df


def generate_pairwise_data(
    df,
    feature_columns,
    max_pairs_per_query=MAX_PAIRS_PER_QUERY,
    random_state=RANDOM_STATE,
):
    rng = np.random.default_rng(random_state)

    feature_chunks = []
    label_chunks = []

    total_possible_pairs = 0
    sampled_pairs = 0

    for qid, group in df.groupby(QUERY_COLUMN, sort=False):

        X = group[feature_columns].to_numpy(dtype=np.float32)
        y = group[RELEVANCE_COLUMN].to_numpy(dtype=int)

        label_groups = {
            label: np.flatnonzero(y == label)
            for label in np.unique(y)
        }

        combinations = []

        labels = sorted(label_groups)

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):

                lower_label = labels[i]
                higher_label = labels[j]

                lower_indices = label_groups[lower_label]
                higher_indices = label_groups[higher_label]

                possible = (
                    len(lower_indices) * len(higher_indices)
                )

                if possible > 0:
                    combinations.append({
                        "lower_indices": lower_indices,
                        "higher_indices": higher_indices,
                        "possible": possible,
                    })

        total = sum(item["possible"] for item in combinations)

        if total == 0:
            continue

        budget = min(max_pairs_per_query, total)

        allocations = [
            int(budget * item["possible"] / total)
            for item in combinations
        ]

        remaining = budget - sum(allocations)

        fractions = [
            budget * item["possible"] / total - allocation
            for item, allocation in zip(combinations, allocations)
        ]

        order = np.argsort(fractions)[::-1]

        for index in order[:remaining]:
            allocations[index] += 1

        for item, number in zip(combinations, allocations):

            if number == 0:
                continue

            lower_indices = item["lower_indices"]
            higher_indices = item["higher_indices"]

            flat_indices = rng.choice(
                item["possible"],
                size=number,
                replace=False,
            )

            lower_positions = (
                flat_indices // len(higher_indices)
            )

            higher_positions = (
                flat_indices % len(higher_indices)
            )

            lower_docs = lower_indices[lower_positions]
            higher_docs = higher_indices[higher_positions]

            # Higher relevance label is preferred.
            differences = (
                X[higher_docs] - X[lower_docs]
            )

            feature_chunks.append(differences)
            feature_chunks.append(-differences)

            label_chunks.append(
                np.ones(number, dtype=np.int32)
            )

            label_chunks.append(
                -np.ones(number, dtype=np.int32)
            )

            sampled_pairs += number

        total_possible_pairs += total

    if not feature_chunks:
        raise ValueError(
            "No unequal-relevance pairs were found."
        )

    X_pairs = np.concatenate(feature_chunks, axis=0)
    y_pairs = np.concatenate(label_chunks, axis=0)

    print("\nPairwise data generation:")
    print(f"Possible unequal pairs: {total_possible_pairs:,}")
    print(f"Sampled document pairs: {sampled_pairs:,}")
    print(f"Training examples: {len(y_pairs):,}")

    return X_pairs, y_pairs


def create_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "svm",
            LinearSVC(
                C=C_VALUE,
                penalty="l2",
                loss="squared_hinge",
                dual=True,
                fit_intercept=False,
                class_weight=None,
                tol=1e-3,
                max_iter=100000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def train_model(model, X_pairs, y_pairs):
    print("\nTraining Ranking SVM...")

    model.fit(X_pairs, y_pairs)

    print("Training completed.")
    print(f"SVM iterations: {model.named_steps['svm'].n_iter_}")

    return model


def calculate_pairwise_accuracy(
    df,
    feature_columns,
    model,
    random_state=RANDOM_STATE + 1,
):
    X_pairs, y_pairs = generate_pairwise_data(
        df,
        feature_columns,
        random_state=random_state,
    )

    pair_scores = model.decision_function(X_pairs)

    outcomes = np.where(
        pair_scores > 0,
        1.0,
        np.where(pair_scores < 0, 0.0, 0.5),
    )

    accuracy = float(np.mean(outcomes))

    print("\nHeld-out pairwise evaluation:")
    print(f"Evaluated pair examples: {len(y_pairs):,}")
    print(f"Pairwise accuracy: {accuracy:.4f}")

    return accuracy


def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances, dtype=int)[:k]

    if len(relevances) == 0:
        return 0.0

    gains = (2 ** relevances) - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))

    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances, k):
    actual_dcg = dcg_at_k(relevances, k)

    ideal_relevances = np.sort(
        np.asarray(relevances, dtype=int)
    )[::-1]

    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_rankings(predictions, k=K):
    results = []

    for qid, group in predictions.groupby(QUERY_COLUMN):

        group = group.sort_values(
            "predicted_score",
            ascending=False,
        )

        relevance = group[RELEVANCE_COLUMN].to_numpy()

        top_k = relevance[:k]

        relevant_count = int(np.sum(relevance > 0))
        retrieved_relevant = int(np.sum(top_k > 0))

        precision = (
            retrieved_relevant / len(top_k)
            if len(top_k) else 0.0
        )

        recall = (
            retrieved_relevant / relevant_count
            if relevant_count else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        ndcg = ndcg_at_k(relevance, k)

        results.append({
            "qid": qid,
            f"precision_at_{k}": precision,
            f"recall_at_{k}": recall,
            f"f1_at_{k}": f1,
            f"ndcg_at_{k}": ndcg,
        })

    results_df = pd.DataFrame(results)

    print("\n========== RANKING EVALUATION ==========")
    print(f"Test queries: {len(results_df)}")
    print(f"Cutoff: {k}")
    print("Relevant documents: relevance > 0")
    print("NDCG gain: 2^relevance - 1")

    print("\nMacro-averaged results:")

    for metric in [
        f"precision_at_{k}",
        f"recall_at_{k}",
        f"f1_at_{k}",
        f"ndcg_at_{k}",
    ]:
        print(f"{metric}: {results_df[metric].mean():.4f}")

    return results_df


def save_feature_weights(model, feature_columns):
    svm = model.named_steps["svm"]
    scaler = model.named_steps["scaler"]

    weights = svm.coef_[0] / scaler.scale_

    weights_df = pd.DataFrame({
        "feature": feature_columns,
        "weight": weights,
        "absolute_weight": np.abs(weights),
    })

    weights_df = weights_df.sort_values(
        "absolute_weight",
        ascending=False,
    )

    weights_df.to_csv(WEIGHTS_PATH, index=False)

    print(f"\nFeature weights saved: {WEIGHTS_PATH}")

    return weights_df


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a pairwise Ranking SVM."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the existing feature dataset.",
    )

    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load the already prepared dataset.
    df, feature_columns = load_dataset(args.data)

    # Split by query to prevent query leakage.
    train_df, test_df = split_by_query(df)

    # Generate training pairs using training queries only.
    X_train_pairs, y_train_pairs = generate_pairwise_data(
        train_df,
        feature_columns,
    )

    # Train the Ranking SVM.
    model = create_model()
    model = train_model(
        model,
        X_train_pairs,
        y_train_pairs,
    )

    # Evaluate pairwise ranking on unseen queries.
    pairwise_accuracy = calculate_pairwise_accuracy(
        test_df,
        feature_columns,
        model,
    )

    # Generate scores for every test query-document row.
    X_test = test_df[feature_columns].to_numpy(dtype=np.float64)

    predictions = test_df[
        [QUERY_COLUMN, DOCUMENT_COLUMN, RELEVANCE_COLUMN]
    ].copy()

    predictions["predicted_score"] = model.decision_function(
        X_test
    )

    predictions = predictions.sort_values(
        [QUERY_COLUMN, "predicted_score"],
        ascending=[True, False],
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    # Evaluate top-k ranking.
    evaluation_results = evaluate_rankings(
        predictions,
        K,
    )

    evaluation_results.to_csv(
        EVALUATION_PATH,
        index=False,
    )

    # Save model and feature weights.
    joblib.dump(model, MODEL_PATH)

    save_feature_weights(
        model,
        feature_columns,
    )

    metadata = {
        "model_type": "Pairwise Linear Ranking SVM",
        "feature_columns": feature_columns,
        "query_column": QUERY_COLUMN,
        "document_column": DOCUMENT_COLUMN,
        "relevance_column": RELEVANCE_COLUMN,
        "relevance_order": "4 > 3 > 2 > 1 > 0",
        "relevant_threshold": 1,
        "C": C_VALUE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "max_pairs_per_query": MAX_PAIRS_PER_QUERY,
        "evaluation_cutoff": K,
        "test_pairwise_accuracy": pairwise_accuracy,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n========== SAVED OUTPUTS ==========")
    print(f"Model: {MODEL_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Evaluation: {EVALUATION_PATH}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Metadata: {METADATA_PATH}")

    print("\nRanking SVM pipeline completed.")


if __name__ == "__main__":
    main()
