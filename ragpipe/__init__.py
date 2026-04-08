"""ragpipe — Context Engineering Platform: production-grade RAG with knowledge graphs, agentic retrieval, self-improving pipelines, and simulation testing."""

__version__ = "3.0.0"

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
