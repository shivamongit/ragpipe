"""Cookbook 01: Basic RAG Pipeline with Ollama (fully local, zero cost).

Shows how to set up a complete RAG pipeline using local models:
- Ollama for embeddings and generation
- NumPy retriever (no external vector DB)
- Token-based chunking

Prerequisites:
    ollama pull nomic-embed-text
    ollama pull gemma4
"""

from ragpipe.chunkers.token import TokenChunker
from ragpipe.core import Document, Pipeline
from ragpipe.embedders.ollama import OllamaEmbedder
from ragpipe.generators.ollama_gen import OllamaGenerator
from ragpipe.retrievers.numpy_retriever import NumpyRetriever

# 1. Build components
chunker = TokenChunker(chunk_size=256, overlap=32)
embedder = OllamaEmbedder(model="nomic-embed-text")
retriever = NumpyRetriever(dim=embedder.dim)
generator = OllamaGenerator(model="gemma4")

# 2. Assemble pipeline
pipeline = Pipeline(
    chunker=chunker,
    embedder=embedder,
    retriever=retriever,
    generator=generator,
    top_k=5,
)

# 3. Ingest documents
docs = [
    Document(
        content="RAG (Retrieval-Augmented Generation) combines information retrieval "
        "with language model generation. It retrieves relevant documents from a "
        "knowledge base and uses them as context for generating accurate answers.",
        metadata={"source": "rag_intro.md"},
    ),
    Document(
        content="Vector databases store embeddings — numerical representations of text. "
        "When a query arrives, it's converted to an embedding and compared against "
        "stored vectors using cosine similarity or other distance metrics.",
        metadata={"source": "vector_db.md"},
    ),
]

stats = pipeline.ingest(docs)
print(f"Ingested: {stats['documents']} docs → {stats['chunks']} chunks")

# 4. Query
result = pipeline.query("How does RAG work?")
print(f"\nAnswer: {result.answer}")
print(f"Model: {result.model} | Tokens: {result.tokens_used} | Latency: {result.latency_ms}ms")
print(f"Sources: {len(result.sources)}")
for s in result.sources:
    print(f"  [{s.rank}] score={s.score:.3f}: {s.chunk.text[:80]}...")
