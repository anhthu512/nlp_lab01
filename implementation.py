"""Part E:
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    """Return the tokens used by all core functions."""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(documents: Iterable[str]) -> list[str]:
    """Build a deterministic, lexicographically sorted vocabulary."""
    return sorted({token for document in documents for token in tokenize(document)})


def compute_counts(documents: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    """Return a document-term count matrix."""
    term_to_index = {term: index for index, term in enumerate(vocabulary)}
    counts = np.zeros((len(documents), len(vocabulary)), dtype=float)
    for row_index, document in enumerate(documents):
        for token in tokenize(document):
            column_index = term_to_index.get(token)
            if column_index is not None:
                counts[row_index, column_index] += 1.0
    return counts


def compute_tf(counts: np.ndarray) -> np.ndarray:
    """Normalize each document's counts by its total token count.

    Empty documents remain all-zero rows.
    """
    counts = np.asarray(counts, dtype=float)
    row_totals = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )


def compute_idf(counts: np.ndarray) -> np.ndarray:
    """Compute smooth inverse document frequency for each term."""
    counts = np.asarray(counts, dtype=float)
    document_frequency = np.count_nonzero(counts > 0, axis=0)
    number_of_documents = counts.shape[0]
    return np.log((1.0 + number_of_documents) / (1.0 + document_frequency)) + 1.0


def compute_tfidf(tf: np.ndarray, idf: np.ndarray) -> np.ndarray:
    """Multiply TF by IDF and L2-normalize each document row."""
    weighted = np.asarray(tf, dtype=float) * np.asarray(idf, dtype=float)
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    return np.divide(weighted, norms, out=np.zeros_like(weighted), where=norms != 0)


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Return cosine similarity, with 0.0 for either zero vector."""
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _run_tests() -> None:
    documents = ["cat eats fish", "dog eats fish", "cat likes fish"]
    vocabulary = build_vocabulary(documents)
    expected_vocabulary = ["cat", "dog", "eats", "fish", "likes"]
    assert vocabulary == expected_vocabulary

    counts = compute_counts(documents, vocabulary)
    expected_counts = np.array([
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1],
    ], dtype=float)
    np.testing.assert_array_equal(counts, expected_counts)

    tf = compute_tf(counts)
    np.testing.assert_allclose(tf.sum(axis=1), np.ones(len(documents)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(compute_tf(np.zeros((1, len(vocabulary)))).sum(), 0.0)

    expected_idf = np.array([
        math.log(4 / 3) + 1,
        math.log(4 / 2) + 1,
        math.log(4 / 3) + 1,
        math.log(4 / 4) + 1,
        math.log(4 / 2) + 1,
    ])
    np.testing.assert_allclose(compute_idf(counts), expected_idf, rtol=0, atol=1e-12)

    idf = compute_idf(counts)
    tfidf = compute_tfidf(tf, idf)
    weighted = tf * idf
    expected_tfidf = weighted / np.linalg.norm(weighted, axis=1, keepdims=True)
    np.testing.assert_allclose(tfidf, expected_tfidf, rtol=0, atol=1e-12)
    assert 0.0 <= cosine_similarity(tfidf[0], tfidf[1]) <= 1.0
    assert cosine_similarity(np.zeros(2), np.ones(2)) == 0.0
    assert abs(cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-12

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    reference_vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        vocabulary=vocabulary,
    )
    reference_counts = reference_vectorizer.transform(documents).toarray()
    reference_transformer = TfidfTransformer(
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False
    )
    reference_tfidf = reference_transformer.fit_transform(reference_counts).toarray()
    reference_tfidf_from_normalized_tf = reference_transformer.fit_transform(tf).toarray()
    np.testing.assert_allclose(counts, reference_counts, rtol=0, atol=1e-12)
    np.testing.assert_allclose(idf, reference_transformer.idf_, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tfidf, reference_tfidf, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tfidf, reference_tfidf_from_normalized_tf, rtol=0, atol=1e-12)

    print("All Part E tests passed.")


if __name__ == "__main__":
    _run_tests()
