"""ragpipe — Production-grade modular RAG framework."""

__version__ = "0.4.0"

from ragpipe.core import Document, Chunk, RetrievalResult, Pipeline
from ragpipe.chunkers.base import BaseChunker
from ragpipe.embedders.base import BaseEmbedder
from ragpipe.retrievers.base import BaseRetriever
from ragpipe.rerankers.base import BaseReranker
from ragpipe.generators.base import BaseGenerator

__all__ = [
    "Document",
    "Chunk",
    "RetrievalResult",
    "Pipeline",
    "BaseChunker",
    "BaseEmbedder",
    "BaseRetriever",
    "BaseReranker",
    "BaseGenerator",
]
