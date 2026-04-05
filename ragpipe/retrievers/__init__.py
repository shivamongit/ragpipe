from ragpipe.retrievers.base import BaseRetriever
from ragpipe.retrievers.numpy_retriever import NumpyRetriever
from ragpipe.retrievers.bm25_retriever import BM25Retriever
from ragpipe.retrievers.hybrid_retriever import HybridRetriever

__all__ = [
    "BaseRetriever",
    "NumpyRetriever",
    "BM25Retriever",
    "HybridRetriever",
]

# Optional retrievers — imported only if their dependencies are installed
def __getattr__(name):
    if name == "FaissRetriever":
        from ragpipe.retrievers.faiss_retriever import FaissRetriever
        return FaissRetriever
    if name == "ChromaRetriever":
        from ragpipe.retrievers.chroma_retriever import ChromaRetriever
        return ChromaRetriever
    if name == "QdrantRetriever":
        from ragpipe.retrievers.qdrant_retriever import QdrantRetriever
        return QdrantRetriever
    raise AttributeError(f"module 'ragpipe.retrievers' has no attribute {name!r}")
