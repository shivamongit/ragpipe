"""Tests for evaluation metrics."""

from ragpipe.core import Chunk, RetrievalResult
from ragpipe.evaluation.metrics import (
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    faithfulness_score,
)


def _make_results(doc_ids: list[str]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=Chunk(text=f"text from {did}", doc_id=did, chunk_index=0),
            score=1.0 - i * 0.1,
            rank=i + 1,
        )
        for i, did in enumerate(doc_ids)
    ]


def test_hit_rate_found():
    results = _make_results(["a", "b", "c"])
    assert hit_rate(results, {"b"}) == 1.0


def test_hit_rate_not_found():
    results = _make_results(["a", "b", "c"])
    assert hit_rate(results, {"z"}) == 0.0


def test_mrr_first():
    results = _make_results(["a", "b", "c"])
    assert mrr(results, {"a"}) == 1.0


def test_mrr_second():
    results = _make_results(["a", "b", "c"])
    assert mrr(results, {"b"}) == 0.5


def test_mrr_not_found():
    results = _make_results(["a", "b", "c"])
    assert mrr(results, {"z"}) == 0.0


def test_precision_at_k():
    results = _make_results(["a", "b", "c", "d", "e"])
    assert precision_at_k(results, {"a", "c"}, k=5) == 0.4
    assert precision_at_k(results, {"a", "b"}, k=2) == 1.0


def test_recall_at_k():
    results = _make_results(["a", "b", "c"])
    assert recall_at_k(results, {"a", "b", "d"}, k=3) == 2.0 / 3.0


def test_ndcg_at_k_perfect():
    results = _make_results(["a", "b"])
    score = ndcg_at_k(results, {"a", "b"}, k=2)
    assert abs(score - 1.0) < 0.001


def test_ndcg_at_k_partial():
    results = _make_results(["x", "a"])
    score = ndcg_at_k(results, {"a"}, k=2)
    assert 0 < score < 1.0


def test_faithfulness_score_full_overlap():
    answer = "the cat sat on the mat"
    sources = ["the cat sat on the mat"]
    scores = faithfulness_score(answer, sources)
    assert scores["unigram_overlap"] == 1.0


def test_faithfulness_score_no_overlap():
    answer = "completely different words here"
    sources = ["nothing matches at all"]
    scores = faithfulness_score(answer, sources)
    assert scores["unigram_overlap"] < 0.5


def test_faithfulness_score_empty():
    scores = faithfulness_score("", ["some source"])
    assert scores["unigram_overlap"] == 0.0
