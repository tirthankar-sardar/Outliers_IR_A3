# Outliers_IR_A3 – Project Setup and Execution Guide

## 1. Prerequisites

The project uses Python and standard scientific-computing libraries.

| Requirement | Version / Details |
|---|---|
| Python | 3.9 or newer |
| pip | Latest recommended |
| NumPy | Required for feature matrix construction |
| Pandas | Required for dataset preprocessing |
| NLTK | Required for tokenization and stemming |
| scikit-learn | Required for SVM training and feature scaling |
| pytest | Required for running tests |
| Git | Required to clone the repository |

**Recommended environment:** Python 3.11 with a virtual environment.

Python 3.9+ is specified by the preprocessing, feature extraction and feature matrix documentation. Python 3.11 is a suggested common environment, not a separately documented tested version.

## 2. Clone the Repository

Open a terminal and execute:

```bash
git clone https://github.com/trishitasaha/Outliers_IR_A3.git

cd Outliers_IR_A3
```

The repository contains separate modules for dataset preparation, Boolean Weighted Zone Scoring, feature extraction and feature matrix construction.

## 3. Create a Virtual Environment

It is recommended to create a virtual environment to avoid conflicts with libraries installed in the system Python environment.

### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

## 4. Install Required Libraries

Install the libraries required by the project:

```bash
pip install numpy pandas nltk scikit-learn pytest
```

The main dependencies are:

- **NumPy:** Numerical operations and feature matrix construction.
- **Pandas:** Processing and handling the Cranfield dataset CSV files.
- **NLTK:** Stopword removal and Porter stemming.
- **Scikit-learn:** Feature standardization and Linear SVM implementation.
- **Pytest:** Unit testing and validation of individual modules.

The feature extraction module itself uses only the Python standard library. The feature matrix module additionally requires NumPy and pytest for its documented execution and tests.

## 5. Run the Project in the Correct Order

**How to Run (`outliers_testA.py` – Boolean Weighted Query Tester)**

Run the query tester from the project root:

```bash
python3 boolean_weighted/outliers_testA.py
```

Or move into `boolean_weighted` first and run:

```bash
cd boolean_weighted
python3 outliers_testA.py
```

The script prompts for one query at a time. Enter a query at `Query>`; type `quit`, `exit`, or `q` to stop. Queries are normalized, Porter-stemmed, and filtered using the collection's stopwords before matching.

The tester returns up to **10 documents** by default. The current tuned configuration requires at least **3 distinct query-term matches in the title** and uses title weight 1 (all other field weights are 0). Results are ordered by descending score, then ascending document ID. The script prints each document's ID, score, title, and a body snippet; it does not write results to an output file.

Before running, make sure the processed collection exists at `outliers_preprocesing_feature_extraction/outliers_preprocessed/outliers_documents_processed.csv` and the stopword file exists at `outliers_preprocesing_feature_extraction/outliers_raw/stopwords.txt`. If the processed documents are missing, run the preprocessing script first:

```bash
python3 outliers_preprocesing_feature_extraction/outliers_preprocess.py
```

This script is interactive and does not currently accept `--query`, `--query-file`, `--top-k`, or `--output` options.

**How to Run (`outliers_testB.py` – Ranking SVM)**

Run the Ranking SVM-based query tester from the project root:

```bash
python3 ranking_svm/outliers_testB.py
```

Or move into `ranking_svm` first and run:

```bash
cd ranking_svm
python3 outliers_testB.py
```

The script supports both interactive and command-line execution. For a single query, enter the query when prompted or provide it directly using `--query`:

```bash
python3 outliers_testB.py --query "What are the effects of air pollution?"
```

By default, the system retrieves the **top 10 documents**. To change the number of retrieved documents, use `--top-k`:

```bash
python3 outliers_testB.py --query "air pollution" --top-k 20
```

For multiple queries, create a `queries.txt` file containing one query per line and run:

```bash
python3 outliers_testB.py --query-file queries.txt --top-k 10
```

The retrieved document IDs are saved to `svm_query_results.txt` by default. To specify a custom output file, use the `--output` option:

```bash
python3 outliers_testB.py --query "air pollution" --top-k 10 --output results.txt
```

Results are ranked in descending order of Ranking SVM decision scores. The output file contains the retrieved document IDs in ranked order.

## 6. Verify Installation

After installation, verify the Python environment and dependencies:

```bash
python --version

pip --version

python -c "import numpy, pandas, nltk, sklearn, pytest; print('All libraries imported successfully')"
```

Run the preprocessing tests:

```bash
python -m pytest
```

Run the feature extraction and matrix tests from the repository root:

```bash
python3 -m pytest feature_matrix feature_extraction -q
```

For the Boolean module, run:

```bash
cd boolean_weighted
python3 test_boolean.py
```

The Boolean implementation uses only Python's standard library and does not require third-party packages.

## 7. Project Execution Flow

This diagram summarizes the complete workflow.

```text
                    Cranfield Dataset
          Documents, queries and relevance judgments
                              |
                              v
                  Part 1: Dataset Preparation
             Cleaning, tokenization, stemming and splits
                              |
                 +------------+------------+
                 |                         |
                 v                         v
       Part 2: Boolean Zone       Part 3: Feature Extraction
             Scoring                    (22 features)
            Method A                     Method B
                                           |
                                           v
                              Part 3b: Feature Matrix
                                  Construction
                                           |
                                           v
                              Part 4: Linear SVM
                              Training and Ranking
```

## 8. Important Notes

- Always run dataset preparation before running the other modules.
- Do not manually modify the generated processed CSV files unless required.
- Use the same Python virtual environment for all modules.
- The train/validation/test split is shared across the project: 157, 34 and 34 queries, respectively.
- Do not use test data for weight tuning or model selection.
- The feature extraction package does not train the SVM.
- Feature matrix construction does not train the SVM either.
- Each module has its own README containing detailed implementation and execution instructions.

# Outliers_IR_A3

- [Part 1: dataset preparation](preprocessing/README.md)
- [Part 2: Boolean Weighted Zone Scoring](boolean_weighted/README_A.md)
- [Part 3: extraction of the 22 query–document features](feature_extraction/README.md)
- [Part 3b: feature matrix](feature_matrix/README.md)

  
The feature extraction package consumes the existing processed CSV files. Matrix
construction, relevance labeling and SVM training are separate tasks.
