# Information Retrieval – Assignment III
## Method B: Learning to Rank Using Linear SVM

### 1. Project Overview

This project implements a supervised Learning-to-Rank approach using a Pairwise Linear Support Vector Machine (SVM) on the Cranfield Information Retrieval collection.

The model learns document ranking preferences from query-document features and relevance judgments. Its performance is compared against BM25 and Vector Space Model (VSM) baselines.

### 2. Dataset

| Description | Value |
|---|---:|
| Collection | Cranfield |
| Total queries | 225 |
| Total documents | 1,400 |
| Query-document pairs | 315,000 |
| Features per pair | 22 |
| Training queries | 180 (80%) |
| Test queries | 45 (20%) |
| Training rows | 252,000 |
| Test rows | 63,000 |

The dataset contains 22 features for each query-document pair, as specified in the assignment.

### 3. Relevance Mapping

The original relevance judgments were mapped as follows:

| Original qrels label | Mapped relevance |
|---:|---:|
| 1 | 4 |
| 2 | 3 |
| 3 | 2 |
| 4 | 1 |
| -1 | 0 |
| Unjudged document-query pair | 0 |

A higher mapped relevance value indicates a more relevant document. Unjudged pairs are treated as relevance 0 in this experiment.

### 4. Feature Set

The model uses the following 22 features:

1. Query term number
2. Query term ratio
3. Stream length
4. IDF
5. TF sum
6. TF minimum
7. TF maximum
8. TF mean
9. TF variance
10. Normalized TF sum
11. Normalized TF minimum
12. Normalized TF maximum
13. Normalized TF mean
14. Normalized TF variance
15. TF-IDF sum
16. TF-IDF minimum
17. TF-IDF maximum
18. TF-IDF mean
19. TF-IDF variance
20. Boolean model score
21. Vector Space Model score
22. BM25 score

### 5. Implementation Methodology

The following steps were performed:

1. Prepared query-document feature vectors using the 22 specified features.
2. Divided the dataset into training and test sets using an 80:20 query-level split.
3. Generated pairwise training examples from documents having different relevance labels.
4. Used feature differences between document pairs to learn ranking preferences.
5. Applied StandardScaler for feature normalisation.
6. Trained a linear SVM using LinearSVC.
7. Used five-fold GroupKFold cross-validation to select the regularisation parameter C.
8. Ranked test documents using the learned SVM scores.

The model was trained using pairwise preferences, where a document with a higher relevance label is preferred over one with a lower relevance label.

### 6. Hyperparameter Tuning

The regularisation parameter C was selected using five-fold GroupKFold cross-validation. Mean NDCG@10 was used as the primary selection metric.

| C | Mean Validation NDCG@10 |
|---:|---:|
| 0.0001 | 0.3503 |
| **0.0003** | **0.3537** |
| 0.001 | 0.3514 |
| 0.003 | 0.3490 |
| 0.01 | 0.3493 |
| 0.1 | 0.3488 |
| 1 | 0.3489 |
| 10 | 0.3490 |
| 100 | 0.3489 |

**Selected hyperparameter: C = 0.0003**

### 7. Final Test Results

The selected model was evaluated on 45 held-out queries.

#### Evaluation at K = 5

| Method | Precision@5 | Recall@5 | F1@5 | NDCG@5 |
|---|---:|---:|---:|---:|
| Ranking SVM | 0.3733 | 0.3542 | 0.3243 | 0.3974 |
| BM25 | 0.3733 | **0.3681** | **0.3348** | **0.4175** |
| VSM | **0.3911** | 0.3600 | 0.3344 | 0.3930 |

#### Evaluation at K = 10

| Method | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Ranking SVM | **0.2756** | **0.4692** | **0.3124** | 0.4329 |
| BM25 | **0.2756** | 0.4571 | 0.3094 | **0.4488** |
| VSM | 0.2644 | 0.4623 | 0.3033 | 0.4066 |

### 8. Statistical Significance Testing

Paired Wilcoxon signed-rank tests were performed to compare Ranking SVM against BM25 using per-query NDCG values.

| Metric | Mean Difference (SVM − BM25) | p-value |
|---|---:|---:|
| NDCG@5 | -0.0200 | 0.8130 |
| NDCG@10 | -0.0159 | 0.6487 |

Both p-values are greater than 0.05. Therefore, the experiment does not establish a statistically significant difference between Ranking SVM and BM25.

### 9. Feature Importance

The learned standardised coefficients indicate the relative contribution of the features to the ranking function.

The five most influential features were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Normalized TF sum | 10.89% |
| 2 | TF-IDF maximum | 9.92% |
| 3 | BM25 | 9.87% |
| 4 | TF-IDF mean | 9.28% |
| 5 | Query term ratio | 8.86% |

The model combines multiple retrieval signals rather than depending entirely on a single feature.

### 10. Conclusion

The Pairwise Linear Ranking SVM achieved competitive retrieval performance compared with BM25 and VSM.

At K=10, Ranking SVM achieved the highest recall (0.4692) and F1 score (0.3124), while BM25 achieved the highest NDCG (0.4488). The statistical tests did not establish a significant difference between Ranking SVM and BM25.

The experiment demonstrates the effectiveness of supervised pairwise learning for combining multiple query-document features to produce ranked document lists.

### 11. Output Files

The experiment generates the following outputs:

- Trained Ranking SVM model.
- Feature coefficient and importance table.
- Ranked test-query predictions.
- Evaluation results.
- Hyperparameter tuning results.
- Experiment metadata.

The saved model and scaler can be reused to rank documents for unseen queries without requiring relevance judgments.