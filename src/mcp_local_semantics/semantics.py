"""The maths the tools are made of: pure NumPy, no model, no network.

Everything here takes vectors and returns numbers. The encoder lives behind an
interface in encoder.py and is injected, so this module is fully unit-testable
without downloading 90 MB of weights — which is why the test suite runs in
under a second and CI does not depend on a model host being reachable.

That separation is the same one that made an agent loop's iteration cap
testable against a mocked model client:
https://ketanshukla.dev/blog/unit-test-an-iteration-cap-without-a-model
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Match:
    index: int
    score: float
    text: str


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product is a cosine.

    Zero rows are left alone rather than producing NaN. An empty string is a
    legitimate input and should score zero against everything, not poison the
    whole matrix.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between every row of a and every row of b."""
    return l2_normalize(a) @ l2_normalize(b).T


def top_k(scores: np.ndarray, texts: list[str], k: int) -> list[Match]:
    """The best k by score, highest first.

    No similarity floor. A fixed cutoff is not scale-invariant across queries —
    it is too strict for short questions and too loose for long ones, and when
    it is too strict it fails silently. Callers that want a floor can apply one
    to the scores they get back, having seen them.
    https://ketanshukla.dev/blog/deleting-the-similarity-threshold
    """
    if k <= 0 or scores.size == 0:
        return []
    k = min(k, scores.size)
    # argpartition is O(n) where a full sort is O(n log n); only the k winners
    # need ordering afterwards. Matters once a corpus is large enough to be
    # worth searching.
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [Match(int(i), float(scores[i]), texts[i]) for i in idx]


def mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    texts: list[str],
    k: int,
    diversity: float = 0.3,
) -> list[Match]:
    """Maximal Marginal Relevance: relevant results that are not near-duplicates.

    Plain top-k over a corpus containing three paraphrases of the same passage
    returns all three and tells the model one thing. MMR penalises a candidate
    by its similarity to what has already been selected, so the k results cover
    more ground.

    `diversity` at 0 is exactly top-k; at 1 it ignores the query entirely and
    just spreads out. The useful range is narrow and worth measuring rather
    than guessing — see scripts/bench.py.
    """
    if k <= 0 or len(texts) == 0:
        return []

    doc_vecs = l2_normalize(doc_vecs)
    query_vec = l2_normalize(query_vec)[0]
    relevance = doc_vecs @ query_vec
    pairwise = doc_vecs @ doc_vecs.T

    selected: list[int] = []
    remaining = list(range(len(texts)))

    while remaining and len(selected) < min(k, len(texts)):
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best = max(
                remaining,
                key=lambda i: (1 - diversity) * relevance[i]
                - diversity * max(pairwise[i][j] for j in selected),
            )
        selected.append(best)
        remaining.remove(best)

    return [Match(i, float(relevance[i]), texts[i]) for i in selected]


def cluster(
    vectors: np.ndarray, k: int, seed: int = 0, iterations: int = 50
) -> list[int]:
    """K-means with k-means++ seeding, over cosine distance on unit vectors.

    Written out rather than imported so the tool has no scikit-learn
    dependency at runtime and its behaviour is inspectable. Deterministic given
    a seed, because a clustering tool an agent calls twice on the same input
    and gets two answers from is a tool the agent cannot reason about.
    """
    vectors = l2_normalize(vectors)
    n = len(vectors)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    rng = np.random.default_rng(seed)

    # k-means++: first centre at random, each subsequent one chosen with
    # probability proportional to its squared distance from the nearest centre.
    # Uniform seeding routinely puts two centres inside one cluster and leaves
    # another unclaimed.
    centres = [vectors[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min(
            [np.sum((vectors - c) ** 2, axis=1) for c in centres], axis=0
        )
        total = d.sum()
        probs = d / total if total > 0 else np.full(n, 1 / n)
        centres.append(vectors[rng.choice(n, p=probs)])

    centroids = np.array(centres)
    labels = np.zeros(n, dtype=int)

    for _ in range(iterations):
        new_labels = np.argmax(vectors @ centroids.T, axis=1)
        if np.array_equal(new_labels, labels):
            break  # converged
        labels = new_labels
        for j in range(k):
            members = vectors[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
        centroids = l2_normalize(centroids)

    return [int(x) for x in labels]


def classify(
    text_vecs: np.ndarray, label_vecs: np.ndarray, labels: list[str]
) -> list[tuple[str, float]]:
    """Zero-shot classification by nearest label embedding.

    No training and no fixed label set: the labels are supplied per call and
    embedded on the spot, which is what makes this useful as an agent tool
    rather than a fixed classifier.
    """
    scores = cosine_matrix(text_vecs, label_vecs)
    return [
        (labels[int(row.argmax())], float(row.max())) for row in np.atleast_2d(scores)
    ]
