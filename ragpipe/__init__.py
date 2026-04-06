"""ragpipe — Production-grade modular RAG framework with hybrid search, contextual chunking, and query expansion."""

__version__ = "2.2.0"

from ragpipe.core import Document, Chunk, RetrievalResult, GenerationResult, Pipeline
from ragpipe.chunkers.base import BaseChunker
from ragpipe.embedders.base import BaseEmbedder
from ragpipe.retrievers.base import BaseRetriever
from ragpipe.rerankers.base import BaseReranker
from ragpipe.generators.base import BaseGenerator

__all__ = [
    "Document",
    "Chunk",
    "RetrievalResult",
    "GenerationResult",
    "Pipeline",
    "BaseChunker",
    "BaseEmbedder",
    "BaseRetriever",
    "BaseReranker",
    "BaseGenerator",
]
