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

## 2. Project Execution Flow

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

## 3. Clone the Repository

Open a terminal and execute:

```bash
git clone https://github.com/trishitasaha/Outliers_IR_A3.git

cd Outliers_IR_A3
```

The repository contains separate modules for dataset preparation, Boolean Weighted Zone Scoring, feature extraction and feature matrix construction.

## 4. Create a Virtual Environment

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

## 5. Install Required Libraries

Install the libraries required by the project:

```bash
pip install requirements.txt
```

## 6. Run the Project in the Correct Order

### How to Run (`outliers_testA.py` – Boolean Weighted Query Tester)

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

### How to Run (Method B – Ranking SVM)

**Run with a Single Query**

Run the script and enter a query when prompted:

```bash
python outliers_testB.py
```

Alternatively, provide the query directly:

```bash
python outliers_testB.py --query "Aircraft structural design"
```

By default, the system retrieves the **top 10 documents**. To change the number of retrieved documents, use `--top-k`:

```bash
python outliers_testB.py --query "Aircraft structural design" --top-k 20
```

**Run Multiple Queries**

Create a text file named `queries.txt`, containing one query per line:

```text
Aircraft structural design
Heat transfer in fluids
```

Run the script using:

```bash
python outliers_testB.py --query-file queries.txt --top-k 10
```

**Specify Output File**

By default, the retrieved document IDs are saved to `svm_query_results.txt`.

To specify a custom output file for a single query:

```bash
python outliers_testB.py --query "Aircraft structural design" --top-k 10 --output results.txt
```

For multiple queries:

```bash
python outliers_testB.py --query-file queries.txt --top-k 10 --output results.txt
```

**Note:** Results are ranked in descending order of Ranking SVM decision scores. The output file contains the retrieved document IDs in ranked order.

## 7. Verify Installation

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
