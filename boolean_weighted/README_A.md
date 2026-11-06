# Method A: Boolean and Soft Weighted Zone Scoring

This document explains the Boolean Weighted Zone Scoring implementation, the Soft Weighted Zone Scoring extension, and how to run both experiments.

## 1. Project layout and data requirements

Run the commands from the `boolean_weighted` directory. The Boolean experiment reads the Outliers processed documents and queries, and the Cranfield relevance judgments:

```text
Outliers_IR_A3/
├── boolean_weighted/
│   ├── outliers_grid_search.py
│   ├── outliers_boolean_matching.py
│   ├── outliers_scoring.py
│   └── outliers_evaluation.py
├── outliers_preprocesing_feature_extraction/
│   ├── outliers_preprocessed/
│   │   ├── outliers_documents_processed.csv
│   │   └── outliers_queries_processed.csv
│   └── outliers_raw/cranqrel
```

The Boolean grid search uses the processed CSVs and qrels paths relative to the project root. It creates `boolean_weighted/results/boolean_grid_search_results.json`.

---

## 2. Boolean Weighted Zone Scoring

### Objective

Rank documents by considering four document fields: `title`, `author`, `bibliography`, and `body`. Each field receives a Boolean match score, and field weights determine its contribution to the final document score.

### Boolean matching

For a query `q`, document `d`, and field `f`, the implementation counts the number of **distinct** query terms that also occur in that field. A field receives 1 when the count reaches the selected minimum-match threshold; otherwise it receives 0:

```text
B_f(q,d) = 1, if distinct matching query terms >= threshold
           0, otherwise
```

The original OR rule is threshold 1. The experiment compares thresholds 1, 2, and 3.

### Weighted score

The final score is the weighted sum of the four Boolean field scores:

```text
Score(q,d) = w_title * B_title
           + w_author * B_author
           + w_bibliography * B_bibliography
           + w_body * B_body
```

Only documents with a positive score are returned. Results are sorted by descending Boolean score, then descending weighted term-match count, with ascending document ID as the final tie-break.

### Weight learning and evaluation

`outliers_grid_search.py` searches field weights from 0 to 5, excluding the all-zero combination (1,295 combinations). For each threshold (1, 2, and 3), it chooses the weights with the best training F1 at cutoff 10. It then evaluates those weights on validation queries and selects the threshold/configuration with the best validation F1. After this selection is frozen, it evaluates the selected configuration on the test queries. Documents are ranked first by their weighted Boolean score, then by weighted distinct query-term matches, then by document ID. This gives a useful ordering when multiple documents receive the same Boolean score.

Recorded result from the current run:

```text
Selected threshold: 1
Weights: title=1, author=0, bibliography=0, body=1
Training F1:   0.24767
Validation F1: 0.21457
Test precision@10: 0.20882
Test recall@10:    0.32140
Test F1@10:        0.23713
```

### Main Boolean files

- `outliers_boolean_matching.py`: parses token lists and calculates the 0/1 field-match scores.
- `outliers_scoring.py`: combines the four Boolean field scores using their weights.
- `outliers_evaluation.py`: calculates macro precision, recall, and F1 at the selected cutoff. Precision uses a fixed denominator of 10.
- `outliers_grid_search.py`: precomputes field match counts, searches the weights, chooses a threshold using validation performance, and evaluates the selected configuration on test queries.

---

## 3. Soft Weighted Zone Scoring experiment

The soft model is a separate extension. It does not replace the Boolean implementation. Instead of assigning only 0 or 1 to a field, it measures the fraction of distinct query terms found in that field.

For query term set `Q` and field term set `F`:

```text
SoftMatch_f(q,d) = |Q intersection F| / |Q|
```

The final document score is the weighted sum of these fractional field scores:

```text
SoftScore(q,d) = sum over fields f of (w_f * SoftMatch_f(q,d))
```

### Soft experiment files

- `soft_matching.py`: computes the fractional query-term coverage for each field.
- `soft_scoring.py`: calculates the weighted soft score and ranks documents.
- `soft_grid_search.py`: precomputes the field-coverage scores, searches non-zero field-weight combinations from 0 to 5 using training queries, reports validation metrics, and reports the number of adjacent tied-score pairs in the validation top 10.
- `soft_test.py` (if used): evaluates the frozen soft weights on the test queries separately. It saves ranked results and test metrics in a `results/` subdirectory.

Recorded soft experiment results:

```text
Best raw weights:        title=4, author=0, bibliography=0, body=3
Normalized weights:     title=0.5714, author=0, bibliography=0, body=0.4286
Training F1:             0.22909
Validation F1:           0.26139
Separate test F1:        0.22577
```

The separate `soft_test.py` evaluation used the fixed weights above and did not tune them using test relevance judgments.

---

## 4. Procedure to run the Boolean experiment

1. Open Terminal and move into the project folder, then the Boolean implementation directory:

   ```bash
   cd /path/to/Outliers_IR_A3-main/boolean_weighted
   ```

2. Run the Boolean matching example:

   ```bash
   python3 outliers_test_boolean.py
   ```

   It demonstrates the results for minimum-match thresholds 1, 2, and 3.

3. Run weight learning, validation-based threshold selection, and final test evaluation:

   ```bash
   python3 outliers_grid_search.py
   ```

4. (Optional) Run the query/tie diagnostic:

   ```bash
   python3 diagnose_boolean.py
   ```

   The script prints summary statistics and creates `boolean_diagnostic_report.csv` in the current directory.

No third-party Python package is required by the Boolean scripts; they use the Python standard library.

---

## 5. Procedure to run the Soft experiment

1. Copy `soft_matching.py`, `soft_scoring.py`, and `soft_grid_search.py` into the existing `boolean_weighted` directory. Keep the original Boolean files unchanged.

2. Install NumPy if it is not already installed:

   ```bash
   python3 -m pip install numpy
   ```

3. **Compatibility adjustment:** the current `grid_search.py` loader returns six values (including test query IDs). In `soft_grid_search.py`, make its `load_data` unpacking match that six-value return. Replace the assignment under `if __name__ == "__main__":` with:

   ```python
   (
       documents,
       query_map,
       relevance,
       train_query_ids,
       val_query_ids,
       _test_query_ids,
   ) = load_data(DATA_DIR)
   ```

   The leading underscore indicates that this script does not use the test query IDs while learning weights or reporting validation metrics.

4. From inside `boolean_weighted`, run:

   ```bash
   python3 soft_grid_search.py
   ```

   This prints the best training weights, normalized weights, training metrics, validation metrics, and validation top-10 tie statistics. This script does **not** evaluate the test set.

5. (Optional) For the separate soft test evaluation, place `soft_test.py` in `boolean_weighted` and run:

   ```bash
   python3 soft_test.py
   ```

   The script uses the frozen weights `title=4, author=0, bibliography=0, body=3`. If you choose a different configuration, update `BEST_WEIGHTS` in `soft_test.py` before running it. It writes test rankings and metrics into `boolean_weighted/results/`.

---

## 6. Evaluation and interpretation notes

- Training queries are used to learn field weights. Validation queries are used to choose among Boolean thresholds/configurations and to report the soft model's validation performance. Test queries are reserved for the final evaluation after the configuration is frozen.
- A document labelled `not judged relevant` in the sample output is not necessarily confirmed non-relevant; it means there is no positive relevance judgment for that query-document pair in the available relevance file.
- Boolean Precision@10 uses a fixed denominator of 10, including when fewer than 10 documents are returned. This matches the standard cutoff metric and allows comparisons with the soft test result.
