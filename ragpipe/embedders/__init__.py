from ragpipe.embedders.base import BaseEmbedder
from ragpipe.embedders.ollama import OllamaEmbedder
from ragpipe.embedders.jina import JinaEmbedder

__all__ = [
    "BaseEmbedder",
    "OllamaEmbedder",
    "JinaEmbedder",
]

# Optional embedders — imported only if their dependencies are installed
def __getattr__(name):
    if name == "OpenAIEmbedder":
        from ragpipe.embedders.openai import OpenAIEmbedder
        return OpenAIEmbedder
    if name == "SentenceTransformerEmbedder":
        from ragpipe.embedders.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder
    if name == "VoyageEmbedder":
        from ragpipe.embedders.voyage import VoyageEmbedder
        return VoyageEmbedder
    raise AttributeError(f"module 'ragpipe.embedders' has no attribute {name!r}")
