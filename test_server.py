"""Test server with mock pipeline for UI demo — no Ollama required."""
import time
import random
import numpy as np
from ragpipe.core import Document, Pipeline, GenerationResult
from ragpipe.chunkers import RecursiveChunker
from ragpipe.embedders.base import BaseEmbedder
from ragpipe.retrievers import NumpyRetriever
from ragpipe.generators.base import BaseGenerator


class MockEmbedder(BaseEmbedder):
    def __init__(self):
        self._dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [np.random.randn(self._dim).tolist() for _ in texts]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    @property
    def dim(self) -> int:
        return self._dim


class MockGenerator(BaseGenerator):
    def generate(self, prompt: str, context: str = "", **kwargs) -> GenerationResult:
        time.sleep(0.2)
        if "key findings" in prompt.lower():
            answer = "Based on the documents, the key findings are:\n\n1. **RAG significantly reduces hallucinations** by grounding LLM responses in retrieved evidence.\n2. **Hybrid retrieval (dense + sparse)** outperforms either method alone, achieving up to 15% better recall.\n3. **Chunk size of 256-512 tokens** provides the best balance between precision and context coverage.\n4. **Semantic caching** can reduce API costs by 60-80% for repeated similar queries."
        elif "summarize" in prompt.lower() or "summary" in prompt.lower():
            answer = "The documents cover several core topics:\n\n- **Retrieval-Augmented Generation (RAG)**: Combining information retrieval with LLM generation for factual, grounded answers.\n- **Vector databases**: FAISS, ChromaDB, and Qdrant for efficient similarity search at scale.\n- **Context engineering**: Programmable context composition replacing naive top-K retrieval.\n- **Evaluation**: Metrics like hit rate, MRR, NDCG, and LLM-as-Judge for measuring pipeline quality."
        elif "compare" in prompt.lower():
            answer = "Comparing the methodologies discussed in the documents:\n\n| Approach | Strengths | Weaknesses |\n|----------|-----------|------------|\n| **Dense retrieval** | Semantic understanding, handles paraphrases | Requires embeddings, slower indexing |\n| **Sparse retrieval (BM25)** | Fast, exact keyword matching | Misses semantic similarity |\n| **Hybrid (RRF)** | Best of both worlds | More complex setup |\n\nThe documents recommend **hybrid retrieval** as the default for production systems."
        elif "recommend" in prompt.lower():
            answer = "The documents list these recommendations:\n\n1. Start with **hybrid retrieval** (dense + BM25) for best recall\n2. Use **recursive chunking** with 512 tokens and 64 overlap\n3. Add **semantic caching** to reduce costs on repeated queries\n4. Implement **answer verification** to catch hallucinations before serving\n5. Run **simulation testing** against adversarial queries before deploying"
        else:
            answer = f"Based on the retrieved context, here is what I found regarding your query:\n\nThe documents contain relevant information that addresses your question. Key themes include retrieval-augmented generation, vector search optimization, and context engineering best practices.\n\nThe evidence suggests that combining multiple retrieval strategies with careful context composition produces the most reliable results for production RAG systems."
        return GenerationResult(
            answer=answer,
            model="mock-llm-v1",
            tokens_used=random.randint(80, 250),
            latency_ms=random.uniform(150, 400),
            sources=[],
        )

    async def agenerate(self, prompt: str, context: str = "", **kwargs) -> GenerationResult:
        return self.generate(prompt, context, **kwargs)

    async def astream(self, prompt: str, context: str = "", **kwargs):
        result = self.generate(prompt, context, **kwargs)
        for word in result.answer.split():
            yield word + " "

    def stream(self, prompt: str, context: str = "", **kwargs):
        result = self.generate(prompt, context, **kwargs)
        for word in result.answer.split():
            yield word + " "


def main():
    import uvicorn
    from ragpipe.server.app import create_app

    pipe = Pipeline(
        chunker=RecursiveChunker(chunk_size=256, overlap=32),
        embedder=MockEmbedder(),
        retriever=NumpyRetriever(),
        generator=MockGenerator(),
    )

    # Pre-load sample documents
    docs = [
        Document(
            content=(
                "RAG (Retrieval-Augmented Generation) is a technique that combines "
                "information retrieval with text generation. It first retrieves relevant "
                "documents from a knowledge base, then uses them as context for an LLM "
                "to generate accurate, grounded answers. RAG reduces hallucinations "
                "and keeps responses factual by anchoring them in real data. Modern RAG "
                "systems use hybrid retrieval combining dense vector search with sparse "
                "BM25 matching for optimal recall."
            ),
            metadata={"source": "rag_overview.md"},
            doc_id="doc-rag-overview",
        ),
        Document(
            content=(
                "Vector databases like FAISS, ChromaDB, and Qdrant store document "
                "embeddings for fast similarity search. FAISS is optimized for speed "
                "with GPU support. ChromaDB offers persistent local storage with metadata "
                "filtering. Qdrant provides scalable cloud deployment. For most use cases, "
                "hybrid retrieval with Reciprocal Rank Fusion (RRF) outperforms pure "
                "dense or sparse retrieval by 10-15% on standard benchmarks."
            ),
            metadata={"source": "vector_databases.md"},
            doc_id="doc-vector-dbs",
        ),
        Document(
            content=(
                "Context engineering is the practice of programmatically composing the "
                "optimal context window for an LLM. Instead of naively stuffing top-K "
                "retrieved chunks, context engineering involves deduplication, relevance "
                "prioritization, token budgeting, and compression. A well-engineered "
                "context window can improve answer quality by 30% while reducing token "
                "costs. Key techniques include density-based prioritization and semantic "
                "deduplication using embedding similarity."
            ),
            metadata={"source": "context_engineering.md"},
            doc_id="doc-context-eng",
        ),
        Document(
            content=(
                "Evaluating RAG pipelines requires multiple metrics across retrieval "
                "and generation quality. Retrieval metrics include hit rate, MRR, "
                "precision@k, recall@k, and NDCG@k. Generation quality is measured "
                "via faithfulness (grounded in context), relevance (answers the question), "
                "and completeness (covers key points). LLM-as-Judge provides automated "
                "scoring across these dimensions on a 0-5 scale."
            ),
            metadata={"source": "evaluation_guide.md"},
            doc_id="doc-evaluation",
        ),
    ]

    stats = pipe.ingest(docs)
    print(f"✅ Pre-loaded {stats['documents']} docs → {stats['chunks']} chunks")

    app = create_app(pipeline=pipe)

    print(f"\n🚀 ragpipe test server running at http://localhost:8000")
    print(f"🖥  Open the UI at http://localhost:3333\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
