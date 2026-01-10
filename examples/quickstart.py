"""Quickstart example — local RAG pipeline with no API keys needed."""

from ragpipe.core import Document, Pipeline
from ragpipe.chunkers.token import TokenChunker
from ragpipe.retrievers.numpy_retriever import NumpyRetriever

# -- Fake embedder for demo (replace with OpenAIEmbedder or SentenceTransformerEmbedder) --
import hashlib
import numpy as np


class DemoEmbedder:
    """Deterministic hash-based embedder for testing without API keys."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._dim).astype(np.float32)
            embeddings.append(vec.tolist())
        return embeddings

    @property
    def dim(self) -> int:
        return self._dim


# -- Fake generator for demo --
class DemoGenerator:
    """Simple extractive generator that returns top chunk text."""

    def generate(self, question, context):
        from ragpipe.generators.base import GenerationOutput

        if not context:
            return GenerationOutput(answer="No context available.")
        answer_parts = []
        for i, r in enumerate(context):
            answer_parts.append(f"[Source {i+1}] (score: {r.score:.3f}): {r.chunk.text[:200]}")
        return GenerationOutput(
            answer="\n".join(answer_parts),
            model="demo-extractive",
            tokens_used=0,
        )


def main():
    # 1. Create documents
    docs = [
        Document(
            content=(
                "Retrieval-Augmented Generation (RAG) combines information retrieval with "
                "language model generation. The retriever finds relevant passages from a "
                "knowledge base, and the generator synthesizes an answer grounded in those "
                "passages. This approach reduces hallucination compared to pure generation."
            ),
            metadata={"source": "rag_overview.txt"},
        ),
        Document(
            content=(
                "FAISS (Facebook AI Similarity Search) is a library for efficient similarity "
                "search of dense vectors. It supports multiple index types including flat (exact), "
                "IVF (inverted file), and HNSW (hierarchical navigable small world). For small "
                "datasets under 100K vectors, IndexFlatIP with L2 normalization provides exact "
                "cosine similarity with sub-millisecond latency."
            ),
            metadata={"source": "faiss_docs.txt"},
        ),
        Document(
            content=(
                "Cross-encoder rerankers process query-document pairs jointly, producing more "
                "accurate relevance scores than bi-encoder similarity. They are typically used "
                "as a second stage after initial retrieval to refine the top-k results. Popular "
                "models include ms-marco-MiniLM and BGE-reranker."
            ),
            metadata={"source": "reranking.txt"},
        ),
    ]

    # 2. Build pipeline
    pipe = Pipeline(
        chunker=TokenChunker(chunk_size=256, overlap=32),
        embedder=DemoEmbedder(dim=64),
        retriever=NumpyRetriever(),
        generator=DemoGenerator(),
        top_k=3,
    )

    # 3. Ingest documents
    stats = pipe.ingest(docs)
    print(f"Ingested {stats['documents']} documents, {stats['chunks']} chunks")
    print(f"Total indexed: {pipe.chunk_count} chunks")

    # 4. Query
    print("\n--- Query 1 ---")
    result = pipe.query("What is RAG and how does it reduce hallucination?")
    print(result.answer)
    print(f"\nLatency: {result.latency_ms:.1f}ms")

    print("\n--- Query 2 ---")
    result = pipe.query("What index type should I use for small datasets in FAISS?")
    print(result.answer)

    print("\n--- Query 3 ---")
    result = pipe.query("How do cross-encoder rerankers work?")
    print(result.answer)

    # 5. Evaluate retrieval
    from ragpipe.evaluation import hit_rate, mrr, precision_at_k

    results = pipe.retrieve("What is FAISS?", top_k=3)
    relevant = {docs[1].doc_id}  # FAISS doc is relevant

    print(f"\n--- Evaluation ---")
    print(f"Hit Rate: {hit_rate(results, relevant)}")
    print(f"MRR: {mrr(results, relevant):.3f}")
    print(f"P@3: {precision_at_k(results, relevant, k=3):.3f}")


if __name__ == "__main__":
    main()
