"""The four tools, as plain functions over an injected encoder.

Kept separate from server.py on purpose: the MCP layer is transport, and this
is the behaviour. Everything here is testable by calling it.

Tool descriptions are written as instructions about *when*, not summaries of
*what*. A vague description means the model never selects the tool at all —
which is the single most consequential line in an MCP server:
https://ketanshukla.dev/blog/an-mcp-server-is-one-route-handler
"""

from __future__ import annotations

from dataclasses import dataclass

from .encoder import Encoder
from .semantics import classify, cluster, cosine_matrix, mmr, top_k

MAX_TEXTS = 512
MAX_CHARS = 8_000


class ToolError(ValueError):
    """Raised for input the caller can fix, with a sentence saying how.

    Returned to the model as an error result rather than thrown at the
    transport, so a bad call becomes a correction on the next turn instead of a
    dead conversation.
    """


def _check(texts: list[str], field: str = "texts") -> list[str]:
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ToolError(f"{field} must be a list of strings")
    if not texts:
        raise ToolError(f"{field} must contain at least one item")
    if len(texts) > MAX_TEXTS:
        raise ToolError(
            f"{field} has {len(texts)} items; the limit is {MAX_TEXTS}. "
            f"Split the batch and call again."
        )
    over = [i for i, t in enumerate(texts) if len(t) > MAX_CHARS]
    if over:
        raise ToolError(
            f"{field}[{over[0]}] is {len(texts[over[0]])} characters; the limit "
            f"is {MAX_CHARS}. Chunk it before embedding."
        )
    return texts


@dataclass
class SemanticTools:
    encoder: Encoder

    # ---------------------------------------------------------------- search
    def semantic_search(
        self, query: str, documents: list[str], k: int = 5, diversity: float = 0.0
    ) -> list[dict]:
        """Rank documents by meaning rather than by shared words."""
        _check(documents, "documents")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")

        doc_vecs = self.encoder.encode(documents)
        query_vec = self.encoder.encode([query])

        if diversity > 0:
            matches = mmr(query_vec, doc_vecs, documents, k, diversity)
        else:
            scores = cosine_matrix(query_vec, doc_vecs)[0]
            matches = top_k(scores, documents, k)

        return [
            {"index": m.index, "score": round(m.score, 4), "text": m.text}
            for m in matches
        ]

    # --------------------------------------------------------------- compare
    def compare_texts(self, texts: list[str]) -> dict:
        """Pairwise similarity, plus the closest and furthest pair."""
        _check(texts)
        if len(texts) < 2:
            raise ToolError("compare_texts needs at least two texts")

        vecs = self.encoder.encode(texts)
        matrix = cosine_matrix(vecs, vecs)

        best = (0, 1, -2.0)
        worst = (0, 1, 2.0)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                s = float(matrix[i][j])
                if s > best[2]:
                    best = (i, j, s)
                if s < worst[2]:
                    worst = (i, j, s)

        return {
            "matrix": [[round(float(v), 4) for v in row] for row in matrix],
            "most_similar": {"a": best[0], "b": best[1], "score": round(best[2], 4)},
            "least_similar": {"a": worst[0], "b": worst[1], "score": round(worst[2], 4)},
        }

    # --------------------------------------------------------------- cluster
    def cluster_texts(self, texts: list[str], groups: int = 3) -> list[dict]:
        """Group texts by topic without being told the topics."""
        _check(texts)
        if groups < 1:
            raise ToolError("groups must be at least 1")

        labels = cluster(self.encoder.encode(texts), groups)
        grouped: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            grouped.setdefault(label, []).append(index)

        return [
            {
                "group": g,
                "indices": members,
                "texts": [texts[i] for i in members],
            }
            for g, members in sorted(grouped.items())
        ]

    # -------------------------------------------------------------- classify
    def classify_texts(self, texts: list[str], labels: list[str]) -> list[dict]:
        """Assign the best-fitting label from a list supplied per call."""
        _check(texts)
        _check(labels, "labels")

        results = classify(
            self.encoder.encode(texts), self.encoder.encode(labels), labels
        )
        return [
            {"index": i, "text": texts[i], "label": label, "score": round(score, 4)}
            for i, (label, score) in enumerate(results)
        ]


# The descriptions the model actually reads. Each names the situations the tool
# applies to, because that is what a selection decision is made from.
TOOL_SPECS = [
    {
        "name": "semantic_search",
        "title": "Search text by meaning",
        "description": (
            "Rank a list of documents by how well each answers a question, using "
            "meaning rather than shared keywords. Use this whenever you have a "
            "pile of text and a question about it: finding the relevant passage "
            "in notes, picking which of several answers fits, or narrowing a "
            "long list before reading it. Runs locally, so it is free to call "
            "and safe to call repeatedly. Set diversity above 0 when the "
            "documents contain near-duplicates and you want coverage rather "
            "than the same passage three times."
        ),
    },
    {
        "name": "compare_texts",
        "title": "Compare texts for similarity",
        "description": (
            "Measure how similar several pieces of text are to one another and "
            "report the closest and furthest pair. Use this to detect duplicate "
            "or near-duplicate entries, check whether two descriptions mean the "
            "same thing, or find the odd one out in a set."
        ),
    },
    {
        "name": "cluster_texts",
        "title": "Group texts by topic",
        "description": (
            "Sort a list of texts into groups by topic without being told what "
            "the topics are. Use this to summarise what themes appear in "
            "feedback, tickets or search results before reading them all, or to "
            "split a long list into coherent batches."
        ),
    },
    {
        "name": "classify_texts",
        "title": "Label texts against categories",
        "description": (
            "Assign each text the best-fitting label from a list of categories "
            "you supply in the call. No training and no fixed label set. Use "
            "this for triage, tagging, sentiment, routing, or any time you have "
            "categories in mind and items to sort into them."
        ),
    },
]
