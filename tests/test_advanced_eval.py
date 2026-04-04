"""Tests for advanced evaluation metrics — MAP@K, ROUGE-L, context precision."""

from ragpipe.core import Chunk, RetrievalResult
from ragpipe.evaluation.metrics import map_at_k, rouge_l, context_precision


def _make_results(doc_ids: list[str]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=Chunk(text=f"text from {did}", doc_id=did, chunk_index=0),
            score=1.0 - i * 0.1,
            rank=i + 1,
        )
        for i, did in enumerate(doc_ids)
    ]


def test_map_at_k_perfect():
    results = _make_results(["a", "b", "c"])
    score = map_at_k(results, {"a", "b"}, k=3)
    assert score == 1.0


def test_map_at_k_partial():
    results = _make_results(["x", "a", "y", "b"])
    score = map_at_k(results, {"a", "b"}, k=4)
    assert 0 < score < 1.0


def test_map_at_k_empty():
    results = _make_results(["x", "y", "z"])
    score = map_at_k(results, {"a"}, k=3)
    assert score == 0.0


def test_rouge_l_identical():
    scores = rouge_l("the cat sat on the mat", "the cat sat on the mat")
    assert scores["f1"] == 1.0


def test_rouge_l_partial():
    scores = rouge_l("the cat sat on the mat", "the cat was on the floor")
    assert 0 < scores["f1"] < 1.0


def test_rouge_l_no_overlap():
    scores = rouge_l("hello world", "goodbye universe")
    assert scores["f1"] == 0.0


def test_rouge_l_empty():
    scores = rouge_l("", "some reference")
    assert scores["f1"] == 0.0


def test_context_precision_perfect():
    results = _make_results(["a", "b", "c"])
    score = context_precision(results, {"a", "b"})
    assert score == 1.0


def test_context_precision_worst():
    results = _make_results(["x", "y", "a"])
    score = context_precision(results, {"a"})
    assert 0 < score < 1.0


def test_context_precision_empty():
    score = context_precision([], {"a"})
    assert score == 0.0
