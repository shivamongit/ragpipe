"""Tests for semantic cache and embedding cache."""

import time

from ragpipe.cache.semantic import SemanticCache
from ragpipe.cache.embedding import EmbeddingCache


# --- SemanticCache ---

def test_semantic_cache_store_and_lookup():
    cache = SemanticCache(threshold=0.95)
    emb = [1.0, 0.0, 0.0]
    cache.store("What is FAISS?", emb, "FAISS is a library by Meta.")
    assert cache.size == 1

    # Exact same embedding should hit
    result = cache.lookup("What is FAISS?", emb)
    assert result == "FAISS is a library by Meta."


def test_semantic_cache_miss_below_threshold():
    cache = SemanticCache(threshold=0.99)
    cache.store("What is FAISS?", [1.0, 0.0, 0.0], "FAISS answer")

    # Orthogonal vector — should miss
    result = cache.lookup("unrelated", [0.0, 1.0, 0.0])
    assert result is None


def test_semantic_cache_similar_hit():
    cache = SemanticCache(threshold=0.90)
    cache.store("What is FAISS?", [1.0, 0.1, 0.0], "FAISS answer")

    # Very similar embedding — should hit
    result = cache.lookup("How does FAISS work?", [0.99, 0.12, 0.01])
    assert result == "FAISS answer"


def test_semantic_cache_ttl_expiry():
    cache = SemanticCache(threshold=0.90, ttl_seconds=0.1)
    emb = [1.0, 0.0, 0.0]
    cache.store("q", emb, "answer")
    assert cache.lookup("q", emb) == "answer"

    time.sleep(0.15)
    assert cache.lookup("q", emb) is None


def test_semantic_cache_max_size():
    cache = SemanticCache(threshold=0.99, max_size=3)
    for i in range(5):
        cache.store(f"q{i}", [float(i), 0.0, 0.0], f"a{i}")
    assert cache.size == 3


def test_semantic_cache_clear():
    cache = SemanticCache()
    cache.store("q", [1.0], "a")
    cache.clear()
    assert cache.size == 0


def test_semantic_cache_stats():
    cache = SemanticCache(threshold=0.90)
    emb = [1.0, 0.0]
    cache.store("q", emb, "a")
    cache.lookup("q", emb)
    stats = cache.stats
    assert stats["entries"] == 1
    assert stats["total_hits"] == 1


# --- EmbeddingCache ---

def test_embedding_cache_put_get():
    cache = EmbeddingCache()
    cache.put(["hello", "world"], [[0.1, 0.2], [0.3, 0.4]])
    result = cache.get(["hello", "world"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_cache_miss():
    cache = EmbeddingCache()
    assert cache.get(["unknown"]) is None


def test_embedding_cache_partial():
    cache = EmbeddingCache()
    cache.put_single("hello", [0.1, 0.2])
    results, missing = cache.get_partial(["hello", "world"])
    assert results[0] == [0.1, 0.2]
    assert results[1] is None
    assert missing == [1]


def test_embedding_cache_lru_eviction():
    cache = EmbeddingCache(max_size=3)
    cache.put(["a", "b", "c"], [[1.0], [2.0], [3.0]])
    cache.put_single("d", [4.0])
    assert cache.size == 3
    # "a" should be evicted (oldest)
    assert cache.get_single("a") is None
    assert cache.get_single("d") == [4.0]


def test_embedding_cache_hit_rate():
    cache = EmbeddingCache()
    cache.put_single("x", [1.0])
    cache.get_single("x")  # hit
    cache.get_single("y")  # miss
    assert cache.hit_rate == 0.5


def test_embedding_cache_clear():
    cache = EmbeddingCache()
    cache.put_single("x", [1.0])
    cache.clear()
    assert cache.size == 0
    assert cache.hit_rate == 0.0
