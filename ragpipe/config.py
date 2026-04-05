"""Declarative pipeline configuration via YAML/dict.

Usage:
    pipe = PipelineConfig.from_yaml("pipeline.yml").build()

Example pipeline.yml:
    chunker:
      type: recursive
      chunk_size: 512
      overlap: 64
    embedder:
      type: ollama
      model: nomic-embed-text
    retriever:
      type: hybrid
      dense: {type: numpy}
      sparse: {type: bm25}
    generator:
      type: ollama
      model: gemma4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragpipe.core import Pipeline


COMPONENT_REGISTRY: dict[str, dict[str, type]] = {}


def _register_defaults():
    """Lazily register built-in components."""
    if COMPONENT_REGISTRY:
        return

    from ragpipe.chunkers.token import TokenChunker
    from ragpipe.chunkers.recursive import RecursiveChunker

    COMPONENT_REGISTRY["chunker"] = {
        "token": TokenChunker,
        "recursive": RecursiveChunker,
    }

    from ragpipe.retrievers.numpy_retriever import NumpyRetriever
    from ragpipe.retrievers.bm25_retriever import BM25Retriever
    from ragpipe.retrievers.hybrid_retriever import HybridRetriever

    COMPONENT_REGISTRY["retriever"] = {
        "numpy": NumpyRetriever,
        "bm25": BM25Retriever,
        "hybrid": HybridRetriever,
    }

    COMPONENT_REGISTRY["generator"] = {}
    COMPONENT_REGISTRY["embedder"] = {}
    COMPONENT_REGISTRY["reranker"] = {}

    # Optional components — register if available
    try:
        from ragpipe.retrievers.faiss_retriever import FaissRetriever
        COMPONENT_REGISTRY["retriever"]["faiss"] = FaissRetriever
    except ImportError:
        pass

    try:
        from ragpipe.retrievers.chroma_retriever import ChromaRetriever
        COMPONENT_REGISTRY["retriever"]["chroma"] = ChromaRetriever
    except ImportError:
        pass

    try:
        from ragpipe.retrievers.qdrant_retriever import QdrantRetriever
        COMPONENT_REGISTRY["retriever"]["qdrant"] = QdrantRetriever
    except ImportError:
        pass

    try:
        from ragpipe.embedders.ollama import OllamaEmbedder
        COMPONENT_REGISTRY["embedder"]["ollama"] = OllamaEmbedder
    except ImportError:
        pass

    try:
        from ragpipe.embedders.sentence_transformer import SentenceTransformerEmbedder
        COMPONENT_REGISTRY["embedder"]["sentence_transformer"] = SentenceTransformerEmbedder
    except ImportError:
        pass

    try:
        from ragpipe.embedders.openai import OpenAIEmbedder
        COMPONENT_REGISTRY["embedder"]["openai"] = OpenAIEmbedder
    except ImportError:
        pass

    try:
        from ragpipe.embedders.voyage import VoyageEmbedder
        COMPONENT_REGISTRY["embedder"]["voyage"] = VoyageEmbedder
    except ImportError:
        pass

    try:
        from ragpipe.embedders.jina import JinaEmbedder
        COMPONENT_REGISTRY["embedder"]["jina"] = JinaEmbedder
    except ImportError:
        pass

    try:
        from ragpipe.generators.ollama_gen import OllamaGenerator
        COMPONENT_REGISTRY["generator"]["ollama"] = OllamaGenerator
    except ImportError:
        pass

    try:
        from ragpipe.generators.openai_gen import OpenAIGenerator
        COMPONENT_REGISTRY["generator"]["openai"] = OpenAIGenerator
    except ImportError:
        pass

    try:
        from ragpipe.generators.anthropic_gen import AnthropicGenerator
        COMPONENT_REGISTRY["generator"]["anthropic"] = AnthropicGenerator
    except ImportError:
        pass

    try:
        from ragpipe.generators.litellm_gen import LiteLLMGenerator
        COMPONENT_REGISTRY["generator"]["litellm"] = LiteLLMGenerator
    except ImportError:
        pass

    try:
        from ragpipe.rerankers.cross_encoder import CrossEncoderReranker
        COMPONENT_REGISTRY["reranker"]["cross_encoder"] = CrossEncoderReranker
    except ImportError:
        pass


def _build_component(category: str, config: dict[str, Any]):
    """Build a single component from config dict."""
    _register_defaults()

    config = dict(config)
    comp_type = config.pop("type")

    registry = COMPONENT_REGISTRY.get(category, {})
    cls = registry.get(comp_type)
    if cls is None:
        available = ", ".join(registry.keys()) or "(none installed)"
        raise ValueError(
            f"Unknown {category} type '{comp_type}'. Available: {available}"
        )

    # Handle nested components (e.g., hybrid retriever's dense/sparse)
    if category == "retriever" and comp_type == "hybrid":
        dense_cfg = config.pop("dense", {"type": "numpy"})
        sparse_cfg = config.pop("sparse", {"type": "bm25"})
        config["dense_retriever"] = _build_component("retriever", dense_cfg)
        config["sparse_retriever"] = _build_component("retriever", sparse_cfg)

    return cls(**config)


@dataclass
class PipelineConfig:
    """Declarative pipeline configuration."""

    chunker: dict[str, Any] = field(default_factory=lambda: {"type": "recursive", "chunk_size": 512, "overlap": 64})
    embedder: dict[str, Any] = field(default_factory=lambda: {"type": "ollama", "model": "nomic-embed-text"})
    retriever: dict[str, Any] = field(default_factory=lambda: {"type": "numpy"})
    generator: dict[str, Any] = field(default_factory=lambda: {"type": "ollama", "model": "gemma4"})
    reranker: dict[str, Any] | None = None
    top_k: int = 5
    rerank_top_k: int = 3

    def build(self) -> Pipeline:
        """Build a Pipeline from this config."""
        chunker = _build_component("chunker", self.chunker)
        embedder = _build_component("embedder", self.embedder)
        retriever = _build_component("retriever", self.retriever)
        generator = _build_component("generator", self.generator)
        reranker = _build_component("reranker", self.reranker) if self.reranker else None

        return Pipeline(
            chunker=chunker,
            embedder=embedder,
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            top_k=self.top_k,
            rerank_top_k=self.rerank_top_k,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create config from a dictionary."""
        defaults = cls()
        return cls(
            chunker=data.get("chunker", defaults.chunker),
            embedder=data.get("embedder", defaults.embedder),
            retriever=data.get("retriever", defaults.retriever),
            generator=data.get("generator", defaults.generator),
            reranker=data.get("reranker"),
            top_k=data.get("top_k", 5),
            rerank_top_k=data.get("rerank_top_k", 3),
        )

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig:
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("Install pyyaml: pip install 'ragpipe[config]'")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        d: dict[str, Any] = {
            "chunker": self.chunker,
            "embedder": self.embedder,
            "retriever": self.retriever,
            "generator": self.generator,
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k,
        }
        if self.reranker:
            d["reranker"] = self.reranker
        return d

    def to_yaml(self, path: str | None = None) -> str:
        """Serialize to YAML string. Optionally write to file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("Install pyyaml: pip install 'ragpipe[config]'")

        text = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

        if path:
            with open(path, "w") as f:
                f.write(text)

        return text
