from __future__ import annotations

import csv
import json
import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

from outliers_porter import PorterStemmer


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "outliers_raw"
PROCESSED_DIR = BASE_DIR / "outliers_preprocessed"

DOCUMENTS_INPUT = RAW_DIR / "cran.all.1400"
QUERIES_INPUT = RAW_DIR / "cran.qry"
STOPWORDS_INPUT = RAW_DIR / "stopwords.txt"

DOCUMENTS_OUTPUT = PROCESSED_DIR / "outliers_documents_processed.csv"
QUERIES_OUTPUT = PROCESSED_DIR / "outliers_queries_processed.csv"

EXPECTED_DOCUMENT_IDS = list(range(1, 1401))
DOCUMENT_ID_RE = re.compile(r"\.I\s+(\d+)")
THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
RAW_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*['’]?|\d+(?:\.\d+)?")
WORD_RE = re.compile(r"[a-z]+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
FIELD_TAGS = {".T", ".A", ".B", ".W"}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    # Preserve numeric value before punctuation becomes a boundary.
    text = THOUSANDS_COMMA_RE.sub("", text)
    return RAW_TOKEN_RE.findall(text)


def normalize(tokens: Iterable[str]) -> list[str]:
    normalized = []
    for token in tokens:
        value = unicodedata.normalize("NFKC", token).lower().replace("’", "'")
        if value.endswith("'s") and value[:-2].isalpha():
            value = value[:-2]
        elif value.endswith("'"):
            value = value[:-1]
        value = value.replace("'", "")
        if WORD_RE.fullmatch(value) or NUMBER_RE.fullmatch(value):
            normalized.append(value)
    return normalized


def stem_tokens(tokens: Iterable[str], stemmer: PorterStemmer) -> list[str]:
    return [stemmer.stem(token) if WORD_RE.fullmatch(token) else token for token in tokens]


def remove_stopwords(tokens: Iterable[str], stemmed_stopwords: set[str]) -> list[str]:
    return [token for token in tokens if token not in stemmed_stopwords]


def load_stemmed_stopwords(path: Path, stemmer: PorterStemmer) -> set[str]:
    words = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            raw_word = line.strip()
            if not raw_word:
                raise ValueError(f"empty stop word at {path}:{line_number}")
            values = normalize(tokenize(raw_word))
            if len(values) != 1 or not WORD_RE.fullmatch(values[0]):
                raise ValueError(f"invalid stop word at {path}:{line_number}: {raw_word!r}")
            word = values[0]
            if word in words:
                raise ValueError(f"duplicate stop word at {path}:{line_number}: {word}")
            words.add(word)

    # Documents and stop words must be compared after the same stemming step.
    return set(stem_tokens(words, stemmer))



def _finish_document(
    doc_id: int,
    fields: dict[str, list[str]],
    seen_fields: set[str],
) -> tuple[int, dict[str, str]]:

    required = {"title", "author", "bibliography", "body"}

    missing = required - seen_fields

    if missing:
        raise ValueError(
            f"document {doc_id} missing fields: {sorted(missing)}"
        )

    return doc_id, {
        name: " ".join(fields[name])
        for name in required
    }


def parse_documents(
    path: Path,
) -> list[tuple[int, dict[str, str]]]:

    documents = []

    tag_to_field = {
        ".T": "title",
        ".A": "author",
        ".B": "bibliography",
        ".W": "body",
    }

    current_id = None
    section = None

    fields = {
        "title": [],
        "author": [],
        "bibliography": [],
        "body": [],
    }

    seen_fields = set()

    with path.open(encoding="ascii") as source:

        for line_number, raw_line in enumerate(source, start=1):

            line = raw_line.rstrip("\r\n")
            marker = line.strip()

            id_match = DOCUMENT_ID_RE.fullmatch(marker)

            if id_match:

                if current_id is not None:
                    documents.append(
                        _finish_document(
                            current_id, fields, seen_fields
                        )
                    )

                current_id = int(id_match.group(1))
                section = None

                fields = {
                    "title": [],
                    "author": [],
                    "bibliography": [],
                    "body": [],
                }

                seen_fields = set()
                continue

            if current_id is None:
                if marker:
                    raise ValueError(
                        f"text before first document at {path}:{line_number}"
                    )
                continue

            if marker in tag_to_field:
                section = tag_to_field[marker]
                seen_fields.add(section)
                continue

            # Preserve stray tag-only lines inside body sections.
            if section == "body" and marker in FIELD_TAGS:
                continue

            if section is not None:
                fields[section].append(line)

    if current_id is not None:
        documents.append(
            _finish_document(current_id, fields, seen_fields)
        )

    document_ids = [doc_id for doc_id, _ in documents]

    if document_ids != EXPECTED_DOCUMENT_IDS:
        raise ValueError(
            "expected sequential document IDs from 1 through 1400"
        )

    return documents


def preprocess_text(
    text: str,
    stemmer: PorterStemmer,
    stemmed_stopwords: set[str],
) -> list[str]:
    tokens = tokenize(text)
    tokens = normalize(tokens)
    tokens = stem_tokens(tokens, stemmer)
    return remove_stopwords(tokens, stemmed_stopwords)



def preprocess_collection(
    input_path: Path,
    stopwords_path: Path,
    output_path: Path,
) -> tuple[int, int]:

    stemmer = PorterStemmer()
    stemmed_stopwords = load_stemmed_stopwords(stopwords_path, stemmer)
    documents = parse_documents(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "title",
        "author",
        "bibliography",
        "body",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)

        writer.writerow([
            "doc_id",
            "title_stemmed_tokens",
            "author_stemmed_tokens",
            "bibliography_stemmed_tokens",
            "body_stemmed_tokens",
        ])

        for doc_id, document_fields in documents:

            processed = {}

            for field in fields:
                processed[field] = preprocess_text(
                    document_fields[field],
                    stemmer,
                    stemmed_stopwords,
                )

            writer.writerow([
                doc_id,
                json.dumps(processed["title"]),
                json.dumps(processed["author"]),
                json.dumps(processed["bibliography"]),
                json.dumps(processed["body"]),
            ])

    return len(documents), len(stemmed_stopwords)


def parse_queries(path: Path) -> list[tuple[int, str]]:
    queries = []
    current_id = None
    query_lines = []
    in_query = False
    next_query_id = 0

    with path.open(encoding="ascii") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")
            marker = line.strip()

            id_match = DOCUMENT_ID_RE.fullmatch(marker)

            if id_match:
                if current_id is not None:
                    queries.append((current_id, " ".join(query_lines)))

                # Assign IDs according to query order, starting at 1.
                next_query_id += 1
                current_id = next_query_id
                query_lines = []
                in_query = False
                continue

            if current_id is None:
                if marker:
                    raise ValueError(
                        f"text before first query at {path}:{line_number}"
                    )
                continue

            if marker == ".W":
                in_query = True
                continue

            if in_query:
                query_lines.append(line)

    if current_id is not None:
        queries.append((current_id, " ".join(query_lines)))

    return queries


def preprocess_queries(
    input_path: Path,
    stopwords_path: Path,
    output_path: Path,
) -> tuple[int, int]:

    stemmer = PorterStemmer()

    stemmed_stopwords = load_stemmed_stopwords(
        stopwords_path, stemmer
    )

    queries = parse_queries(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output:

        writer = csv.writer(output)

        writer.writerow([
            "query_id",
            "query_stemmed_tokens",
        ])

        for query_id, query_text in queries:

            tokens = preprocess_text(
                query_text,
                stemmer,
                stemmed_stopwords,
            )

            writer.writerow([
                query_id,
                json.dumps(tokens),
            ])

    return len(queries), len(stemmed_stopwords)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess Cranfield documents and queries.")
    parser.add_argument("--input",type=Path,default=DOCUMENTS_INPUT,)
    parser.add_argument("--queries",type=Path,default=QUERIES_INPUT,)
    parser.add_argument("--stopwords",type=Path,default=STOPWORDS_INPUT,)
    parser.add_argument("--output",type=Path,default=DOCUMENTS_OUTPUT,)
    parser.add_argument("--query-output",type=Path,default=QUERIES_OUTPUT,)
    return parser.parse_args(argv)


def main(argv=None):

    args = parse_args(argv)

    try:

        document_count, stopword_count = preprocess_collection(
            args.input,
            args.stopwords,
            args.output,
        )

        query_count, _ = preprocess_queries(
            args.queries,
            args.stopwords,
            args.query_output,
        )

    except (OSError, UnicodeError, ValueError) as error:

        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Processed documents: {document_count}")
    print(f"Processed queries: {query_count}")
    print(f"Stopword stems: {stopword_count}")
    print(f"Document output: {args.output}")
    print(f"Query output: {args.query_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
