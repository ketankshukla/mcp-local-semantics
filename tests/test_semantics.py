"""The maths and the tools, exercised through HashEncoder.

No downloads, no torch, no network — so this suite runs in about a second and
CI does not depend on a model host. LocalEncoder is exercised by
scripts/bench.py instead.
"""

import numpy as np
import pytest

from mcp_local_semantics.encoder import HashEncoder
from mcp_local_semantics.semantics import (
    cluster,
    cosine_matrix,
    l2_normalize,
    mmr,
    top_k,
)
from mcp_local_semantics.tools import MAX_TEXTS, SemanticTools, ToolError


@pytest.fixture(scope="module")
def tools():
    return SemanticTools(encoder=HashEncoder())


# --------------------------------------------------------------------------- #
#  Vector maths
# --------------------------------------------------------------------------- #


class TestNormalize:
    def test_rows_become_unit_length(self):
        out = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_zero_row_survives_instead_of_becoming_nan(self):
        """An empty string is a legitimate input. It should score zero against
        everything, not poison the matrix."""
        out = l2_normalize(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert not np.isnan(out).any()

    def test_promotes_a_single_vector_to_a_matrix(self):
        assert l2_normalize(np.array([3.0, 4.0])).shape == (1, 2)


class TestCosine:
    def test_identical_direction_is_one(self):
        a = np.array([[1.0, 2.0, 3.0]])
        assert cosine_matrix(a, a)[0][0] == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        m = cosine_matrix(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
        assert m[0][0] == pytest.approx(0.0)

    def test_magnitude_does_not_matter(self):
        a, b = np.array([[1.0, 1.0]]), np.array([[50.0, 50.0]])
        assert cosine_matrix(a, b)[0][0] == pytest.approx(1.0)


class TestTopK:
    def test_orders_by_score_descending(self):
        matches = top_k(np.array([0.1, 0.9, 0.5]), ["a", "b", "c"], 3)
        assert [m.text for m in matches] == ["b", "c", "a"]

    def test_k_larger_than_the_corpus_is_clamped(self):
        assert len(top_k(np.array([0.5, 0.2]), ["a", "b"], 99)) == 2

    def test_k_of_zero_returns_nothing(self):
        assert top_k(np.array([0.5]), ["a"], 0) == []

    def test_empty_scores_return_nothing(self):
        assert top_k(np.array([]), [], 5) == []

    def test_no_similarity_floor_is_applied(self):
        """Every result is returned regardless of how low it scores. A fixed
        cutoff is not scale-invariant across queries, and when it is too strict
        it fails silently."""
        matches = top_k(np.array([0.01, 0.02]), ["a", "b"], 2)
        assert len(matches) == 2


class TestMMR:
    def test_diversity_zero_matches_plain_top_k(self):
        vecs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        query = np.array([[1.0, 0.0]])
        texts = ["a", "b", "c"]
        plain = [m.index for m in top_k(cosine_matrix(query, vecs)[0], texts, 2)]
        assert [m.index for m in mmr(query, vecs, texts, 2, diversity=0.0)] == plain

    def test_diversity_breaks_up_near_duplicates(self):
        """Two near-identical documents and one different. Plain ranking takes
        both duplicates; MMR should reach for the third."""
        vecs = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        query = np.array([[1.0, 0.05]])
        texts = ["dup1", "dup2", "other"]
        picked = [m.text for m in mmr(query, vecs, texts, 2, diversity=0.8)]
        assert "other" in picked

    def test_k_of_zero_returns_nothing(self):
        assert mmr(np.array([[1.0]]), np.array([[1.0]]), ["a"], 0) == []


class TestCluster:
    def test_separates_two_obvious_groups(self):
        vecs = np.array([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]])
        labels = cluster(vecs, 2, seed=1)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_is_deterministic_for_a_given_seed(self):
        """A tool an agent calls twice on the same input and gets two answers
        from is a tool it cannot reason about."""
        vecs = np.random.default_rng(7).normal(size=(20, 8))
        assert cluster(vecs, 3, seed=42) == cluster(vecs, 3, seed=42)

    def test_more_groups_than_items_is_clamped(self):
        assert len(set(cluster(np.array([[1.0, 0.0], [0.0, 1.0]]), 10))) <= 2

    def test_empty_input_returns_nothing(self):
        assert cluster(np.zeros((0, 4)), 3) == []


# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #


class TestSemanticSearch:
    def test_ranks_the_matching_document_first(self, tools):
        docs = [
            "the deployment runbook covers rollback steps",
            "banana bread recipe with walnuts",
            "invoice line items and proration",
        ]
        out = tools.semantic_search("rollback deployment runbook", docs, k=1)
        assert out[0]["text"] == docs[0]

    def test_respects_k(self, tools):
        out = tools.semantic_search("anything", ["a", "b", "c", "d"], k=2)
        assert len(out) == 2

    def test_results_carry_their_original_index(self, tools):
        out = tools.semantic_search("banana", ["x", "banana bread"], k=1)
        assert out[0]["index"] == 1

    def test_empty_query_is_rejected_with_a_usable_message(self, tools):
        with pytest.raises(ToolError, match="non-empty"):
            tools.semantic_search("   ", ["a"], k=1)

    def test_empty_document_list_is_rejected(self, tools):
        with pytest.raises(ToolError, match="at least one"):
            tools.semantic_search("q", [], k=1)


class TestCompareTexts:
    def test_finds_the_closest_and_furthest_pair(self, tools):
        out = tools.compare_texts(["cat dog", "cat dog", "quantum tensor field"])
        assert {out["most_similar"]["a"], out["most_similar"]["b"]} == {0, 1}
        assert 2 in {out["least_similar"]["a"], out["least_similar"]["b"]}

    def test_matrix_is_square_and_symmetric(self, tools):
        m = tools.compare_texts(["a b", "c d", "e f"])["matrix"]
        assert len(m) == len(m[0]) == 3
        assert m[0][1] == pytest.approx(m[1][0])

    def test_one_text_is_rejected(self, tools):
        with pytest.raises(ToolError, match="at least two"):
            tools.compare_texts(["only one"])


class TestClusterTexts:
    def test_every_text_lands_in_exactly_one_group(self, tools):
        texts = ["alpha beta", "alpha beta gamma", "zulu yankee", "zulu xray"]
        groups = tools.cluster_texts(texts, groups=2)
        placed = sorted(i for g in groups for i in g["indices"])
        assert placed == [0, 1, 2, 3]

    def test_zero_groups_is_rejected(self, tools):
        with pytest.raises(ToolError, match="at least 1"):
            tools.cluster_texts(["a"], groups=0)


class TestClassifyTexts:
    def test_returns_one_row_per_text_with_a_supplied_label(self, tools):
        out = tools.classify_texts(["refund my invoice"], ["billing", "weather"])
        assert len(out) == 1
        assert out[0]["label"] in {"billing", "weather"}

    def test_labels_must_not_be_empty(self, tools):
        with pytest.raises(ToolError, match="labels"):
            tools.classify_texts(["a"], [])


class TestGuardRails:
    def test_batch_over_the_limit_says_what_to_do_about_it(self, tools):
        with pytest.raises(ToolError, match="Split the batch"):
            tools.classify_texts(["x"] * (MAX_TEXTS + 1), ["a"])

    def test_oversized_text_says_what_to_do_about_it(self, tools):
        with pytest.raises(ToolError, match="Chunk it"):
            tools.semantic_search("q", ["x" * 9000], k=1)

    def test_non_string_input_is_rejected(self, tools):
        with pytest.raises(ToolError, match="list of strings"):
            tools.compare_texts([1, 2])  # type: ignore[list-item]


class TestHashEncoder:
    def test_same_text_gives_the_same_vector(self):
        e = HashEncoder()
        assert np.allclose(e.encode(["hello"]), e.encode(["hello"]))

    def test_different_text_gives_a_different_vector(self):
        e = HashEncoder()
        assert not np.allclose(e.encode(["hello"]), e.encode(["goodbye"]))

    def test_empty_batch_returns_the_right_shape(self):
        assert HashEncoder().encode([]).shape == (0, 256)
