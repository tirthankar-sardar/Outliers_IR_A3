from pathlib import Path
import numpy as np
import pandas as pd


# 1. PATHS AND SETTINGS

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "outliers_preprocesing_feature_extraction"
    / "outliers_features"
    / "outliers_svm_training_dataset.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR
    / "outliers_svm_models"
    / "outliers_test_predictions.csv"
)

OUTPUT_DIR = BASE_DIR / "outliers_svm_models"

K = 10

BM25_COLUMN = "bm25"
VSM_COLUMN = "vector_space_model"

# Set this manually if automatic detection does not find
# the SVM score column in your predictions CSV.
SVM_SCORE_COLUMN = None

SVM_SCORE_CANDIDATES = [
    "svm_score",
    "ranking_score",
    "predicted_score",
    "prediction_score",
    "score",
]

# 2. HELPER FUNCTIONS

def find_column(df, possible_names, description):
    for name in possible_names:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find {description}. "
        f"Available columns: {list(df.columns)}"
    )


def calculate_dcg(relevances, k):
    relevances = np.asarray(relevances[:k], dtype=float)

    if len(relevances) == 0:
        return 0.0

    gains = (2 ** relevances) - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))

    return np.sum(gains / discounts)


def evaluate_query(group, score_column, k=10):

    # Rank documents from highest score to lowest score
    ranked = group.sort_values(
        by=score_column,
        ascending=False,
        kind="mergesort"
    )

    top_k = ranked.head(k)

    relevance = ranked["relevance"].to_numpy(dtype=float)
    top_relevance = top_k["relevance"].to_numpy(dtype=float)

    # Binary relevance: relevance > 0
    relevant_top_k = np.sum(top_relevance > 0)
    total_relevant = np.sum(relevance > 0)

    precision = (
        relevant_top_k / len(top_k)
        if len(top_k) > 0 else 0.0
    )

    recall = (
        relevant_top_k / total_relevant
        if total_relevant > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    # Graded relevance: 0, 1, 2, 3, 4
    dcg = calculate_dcg(top_relevance, k)

    ideal_relevance = np.sort(relevance)[::-1]
    idcg = calculate_dcg(ideal_relevance, k)

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision_at_10": precision,
        "recall_at_10": recall,
        "f1_at_10": f1,
        "ndcg_at_10": ndcg,
    }

# 3. LOAD DATA

print("Loading dataset and SVM predictions...")

dataset = pd.read_csv(DATASET_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

# Support alternative column names
dataset_qid = find_column(
    dataset, ["qid", "query_id"], "query ID in dataset"
)

dataset_docid = find_column(
    dataset, ["docid", "doc_id"], "document ID in dataset"
)

dataset_rel = find_column(
    dataset, ["relevance"], "relevance column in dataset"
)

pred_qid = find_column(
    predictions, ["qid", "query_id"], "query ID in predictions"
)

pred_docid = find_column(
    predictions, ["docid", "doc_id"], "document ID in predictions"
)

pred_rel = find_column(
    predictions, ["relevance"], "relevance column in predictions"
)

# Automatically identify the SVM prediction score
if SVM_SCORE_COLUMN is None:
    SVM_SCORE_COLUMN = find_column(
        predictions,
        SVM_SCORE_CANDIDATES,
        "SVM prediction score"
    )

print("SVM score column:", SVM_SCORE_COLUMN)

# 4. PREPARE THE SAME TEST SET FOR ALL MODELS

# Standardise identifier names
dataset = dataset.rename(columns={
    dataset_qid: "qid",
    dataset_docid: "docid",
    dataset_rel: "dataset_relevance",
})

predictions = predictions.rename(columns={
    pred_qid: "qid",
    pred_docid: "docid",
    pred_rel: "relevance",
})

# Convert IDs to strings to avoid mismatched merge types
for df in (dataset, predictions):
    df["qid"] = df["qid"].astype(str)
    df["docid"] = df["docid"].astype(str)

# Check for duplicate query-document pairs
if dataset.duplicated(["qid", "docid"]).any():
    raise ValueError(
        "Dataset contains duplicate qid-docid pairs."
    )

if predictions.duplicated(["qid", "docid"]).any():
    raise ValueError(
        "SVM predictions contain duplicate qid-docid pairs."
    )

# Keep only the required baseline features from the dataset
required_features = [
    "qid",
    "docid",
    BM25_COLUMN,
    VSM_COLUMN,
]

missing_features = [
    col for col in required_features
    if col not in dataset.columns
]

if missing_features:
    raise ValueError(
        f"Missing feature columns in dataset: {missing_features}"
    )

baseline_features = dataset[required_features].copy()

# Merge features onto the exact SVM test documents
comparison_data = predictions[
    ["qid", "docid", "relevance", SVM_SCORE_COLUMN]
].merge(
    baseline_features,
    on=["qid", "docid"],
    how="left",
    validate="one_to_one"
)

# Check that every test document has its baseline features
if comparison_data[[BM25_COLUMN, VSM_COLUMN]].isna().any().any():
    raise ValueError(
        "Some test documents are missing BM25 or VSM features. "
        "Check the qid and docid columns."
    )

# Check that the relevance labels agree
relevance_check = comparison_data.merge(
    dataset[["qid", "docid", "dataset_relevance"]],
    on=["qid", "docid"],
    how="left",
    validate="one_to_one"
)

if not np.array_equal(
    relevance_check["relevance"].to_numpy(),
    relevance_check["dataset_relevance"].to_numpy()
):
    raise ValueError(
        "Relevance labels in predictions do not match the dataset."
    )

# Ensure ranking scores are numeric
score_columns = [
    SVM_SCORE_COLUMN,
    BM25_COLUMN,
    VSM_COLUMN,
]

for col in score_columns:
    comparison_data[col] = pd.to_numeric(
        comparison_data[col],
        errors="coerce"
    )

if comparison_data[score_columns].isna().any().any():
    raise ValueError("Some ranking scores are missing or non-numeric.")

print("Test queries:", comparison_data["qid"].nunique())
print("Test query-document rows:", len(comparison_data))

# 5. EVALUATE ALL THREE MODELS

models = {
    "BM25": BM25_COLUMN,
    "VSM": VSM_COLUMN,
    "Pairwise Linear SVM": SVM_SCORE_COLUMN,
}

per_query_results = []

for model_name, score_column in models.items():

    print(f"Evaluating {model_name}...")

    for qid, group in comparison_data.groupby("qid"):

        metrics = evaluate_query(
            group,
            score_column,
            K
        )

        per_query_results.append({
            "qid": qid,
            "model": model_name,
            **metrics,
        })

# 6. CREATE PER-QUERY COMPARISON

per_query_df = pd.DataFrame(per_query_results)

per_query_path = (
    OUTPUT_DIR / "outliers_per_query_comparison.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

per_query_df.to_csv(
    per_query_path,
    index=False
)

# 7. CREATE MACRO-AVERAGED COMPARISON

metric_columns = [
    "precision_at_10",
    "recall_at_10",
    "f1_at_10",
    "ndcg_at_10",
]

summary_df = (
    per_query_df
    .groupby("model")[metric_columns]
    .mean()
    .reset_index()
)

summary_df.insert(
    1,
    "test_queries",
    comparison_data["qid"].nunique()
)

summary_path = (
    OUTPUT_DIR / "outliers_model_comparison.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

# 8. DISPLAY RESULTS

print("\n" + "=" * 65)
print("RETRIEVAL MODEL COMPARISON")
print("=" * 65)

print(f"Test queries: {comparison_data['qid'].nunique()}")
print(f"Evaluation cutoff: {K}")
print("Binary relevance: relevance > 0")
print("NDCG gain: 2^relevance - 1")

print("\nMacro-averaged results:\n")

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nFiles saved successfully:")
print(summary_path)
print(per_query_path)

print("\nComparison completed.")
