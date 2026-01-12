"""Full RAG pipeline with OpenAI embeddings, FAISS retrieval, and GPT-4o-mini generation.

Requirements:
    pip install 'ragpipe[openai,faiss]'
    export OPENAI_API_KEY=sk-...
"""

from ragpipe.core import Document, Pipeline
from ragpipe.chunkers.recursive import RecursiveChunker
from ragpipe.embedders.openai import OpenAIEmbedder
from ragpipe.retrievers.faiss_retriever import FaissRetriever
from ragpipe.generators.openai_gen import OpenAIGenerator
from ragpipe.loaders.directory import DirectoryLoader


def main():
    # Load documents from a directory
    # loader = DirectoryLoader()
    # docs = loader.load("./my_documents")

    # Or create documents inline
    docs = [
        Document(content="Your document content here...", metadata={"source": "example.txt"}),
    ]

    # Build pipeline
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    pipe = Pipeline(
        chunker=RecursiveChunker(chunk_size=512, overlap=64),
        embedder=embedder,
        retriever=FaissRetriever(dim=embedder.dim, persist_dir="./index"),
        generator=OpenAIGenerator(model="gpt-4o-mini", temperature=0.1),
        top_k=5,
    )

    # Ingest
    stats = pipe.ingest(docs)
    print(f"Ingested: {stats}")

    # Query
    result = pipe.query("What are the key findings?")
    print(f"\nAnswer: {result.answer}")
    print(f"\nSources ({len(result.sources)}):")
    for s in result.sources:
        print(f"  [{s.rank}] score={s.score:.3f} — {s.chunk.text[:100]}...")
    print(f"\nModel: {result.model}, Tokens: {result.tokens_used}, Latency: {result.latency_ms}ms")


if __name__ == "__main__":
    main()
