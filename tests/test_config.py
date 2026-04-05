"""Tests for YAML/dict pipeline configuration."""

import pytest

from ragpipe.config import PipelineConfig, _build_component


def test_build_chunker_token():
    comp = _build_component("chunker", {"type": "token", "chunk_size": 256, "overlap": 32})
    assert comp.__class__.__name__ == "TokenChunker"


def test_build_chunker_recursive():
    comp = _build_component("chunker", {"type": "recursive", "chunk_size": 512, "overlap": 64})
    assert comp.__class__.__name__ == "RecursiveChunker"


def test_build_retriever_numpy():
    comp = _build_component("retriever", {"type": "numpy"})
    assert comp.__class__.__name__ == "NumpyRetriever"


def test_build_retriever_bm25():
    comp = _build_component("retriever", {"type": "bm25"})
    assert comp.__class__.__name__ == "BM25Retriever"


def test_build_retriever_hybrid():
    comp = _build_component("retriever", {
        "type": "hybrid",
        "dense": {"type": "numpy"},
        "sparse": {"type": "bm25"},
    })
    assert comp.__class__.__name__ == "HybridRetriever"


def test_build_unknown_type():
    with pytest.raises(ValueError, match="Unknown chunker type"):
        _build_component("chunker", {"type": "nonexistent"})


def test_config_from_dict():
    data = {
        "chunker": {"type": "token", "chunk_size": 256, "overlap": 32},
        "embedder": {"type": "ollama", "model": "nomic-embed-text"},
        "retriever": {"type": "numpy"},
        "generator": {"type": "ollama", "model": "gemma4"},
        "top_k": 10,
    }
    config = PipelineConfig.from_dict(data)
    assert config.top_k == 10
    assert config.chunker["type"] == "token"


def test_config_to_dict():
    config = PipelineConfig(
        chunker={"type": "recursive", "chunk_size": 512, "overlap": 64},
        embedder={"type": "ollama", "model": "nomic-embed-text"},
        retriever={"type": "numpy"},
        generator={"type": "ollama", "model": "gemma4"},
    )
    d = config.to_dict()
    assert d["chunker"]["type"] == "recursive"
    assert d["generator"]["model"] == "gemma4"
    assert "reranker" not in d


def test_config_build_pipeline():
    config = PipelineConfig(
        chunker={"type": "token", "chunk_size": 256, "overlap": 32},
        embedder={"type": "ollama", "model": "nomic-embed-text"},
        retriever={"type": "numpy"},
        generator={"type": "ollama", "model": "gemma4"},
    )
    pipe = config.build()
    assert pipe.top_k == 5
    assert pipe.chunker.__class__.__name__ == "TokenChunker"
    assert pipe.retriever.__class__.__name__ == "NumpyRetriever"


def test_config_roundtrip_dict():
    original = PipelineConfig(
        chunker={"type": "recursive", "chunk_size": 512, "overlap": 64},
        embedder={"type": "ollama", "model": "nomic-embed-text"},
        retriever={"type": "hybrid", "dense": {"type": "numpy"}, "sparse": {"type": "bm25"}},
        generator={"type": "ollama", "model": "gemma4"},
        top_k=7,
    )
    d = original.to_dict()
    restored = PipelineConfig.from_dict(d)
    assert restored.top_k == 7
    assert restored.retriever["type"] == "hybrid"
