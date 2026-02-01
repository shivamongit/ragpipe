# ragpipe

Production-grade modular RAG framework. Pluggable chunkers, embedders, retrievers, rerankers, and generators — compose your pipeline in 10 lines.

[![CI](https://github.com/shivamongit/ragpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/shivamongit/ragpipe/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Why ragpipe?

Most RAG tutorials glue together OpenAI + FAISS in a single script. That works for demos, not for production. ragpipe gives you:

- **Modular components** — swap chunkers, embedders, retrievers, rerankers, generators independently
- **Multiple chunking strategies** — token-based, recursive, semantic (embedding-similarity breakpoints)
- **Pluggable embedders** — OpenAI, Sentence Transformers (local), or bring your own
- **FAISS + NumPy retrievers** — FAISS for performance, pure NumPy for zero-dependency environments
- **Cross-encoder reranking** — second-stage reranking for precision-critical applications
- **Built-in evaluation** — Hit Rate, MRR, Precision@K, Recall@K, NDCG@K, faithfulness scoring
- **Document loaders** — PDF, DOCX, TXT, Markdown, directory-recursive loading
- **Index persistence** — save and reload FAISS indices across sessions
- **Zero magic** — every component is explicit, typed, and testable

## Install

```bash
pip install ragpipe                          # core (chunkers + numpy retriever)
pip install 'ragpipe[faiss]'                 # + FAISS retriever
pip install 'ragpipe[openai]'                # + OpenAI embedder & generator
pip install 'ragpipe[sentence-transformers]' # + local embedder & reranker
pip install 'ragpipe[all]'                   # everything
```

## Quickstart

```python
from ragpipe import Document, Pipeline
from ragpipe.chunkers import TokenChunker
from ragpipe.embedders import OpenAIEmbedder
from ragpipe.retrievers import FaissRetriever
from ragpipe.generators import OpenAIGenerator

embedder = OpenAIEmbedder(model="text-embedding-3-small")

pipe = Pipeline(
    chunker=TokenChunker(chunk_size=512, overlap=64),
    embedder=embedder,
    retriever=FaissRetriever(dim=embedder.dim, persist_dir="./index"),
    generator=OpenAIGenerator(model="gpt-4o-mini"),
    top_k=5,
)

# Ingest documents
pipe.ingest([
    Document(content="Your document text...", metadata={"source": "report.pdf"}),
])

# Query
result = pipe.query("What are the key findings?")
print(result.answer)
print(result.sources)       # ranked source chunks with scores
print(result.latency_ms)    # end-to-end latency
print(result.tokens_used)   # LLM token consumption
```

## Architecture

```
Document → Loader → Chunker → Embedder → Retriever → Reranker → Generator → Answer
              │                    │           │           │           │
           TextLoader         TokenChunker  OpenAI     FAISS      CrossEncoder  OpenAI
           PDFLoader          Recursive     SentTrans  NumPy                    Custom
           DocxLoader         Semantic      Custom     Custom
           DirectoryLoader    Custom
```

Every component implements a base class. Swap any piece without touching the rest.

## Components

### Chunkers

| Chunker | Strategy | Best for |
|---------|----------|----------|
| `TokenChunker` | Fixed token window + overlap | General purpose, predictable sizes |
| `RecursiveChunker` | Hierarchical separators (¶ → sentence → word) | Preserving semantic structure |
| `SemanticChunker` | Split at embedding similarity breakpoints | Maximum coherence per chunk |

```python
from ragpipe.chunkers import TokenChunker, RecursiveChunker, SemanticChunker

# Token-based: fixed 512-token windows with 64-token overlap
chunker = TokenChunker(chunk_size=512, overlap=64)

# Recursive: tries paragraph → sentence → word boundaries
chunker = RecursiveChunker(chunk_size=512, overlap=64)

# Semantic: splits where consecutive sentence embeddings diverge
chunker = SemanticChunker(embedder=my_embedder, threshold=0.75)
```

### Embedders

| Embedder | Runs locally | Dimensions |
|----------|-------------|------------|
| `OpenAIEmbedder` | No (API) | 1536 / 3072 |
| `SentenceTransformerEmbedder` | Yes | 384 / 768 / 1024 |

```python
from ragpipe.embedders import OpenAIEmbedder, SentenceTransformerEmbedder

# Cloud — high quality, costs money
embedder = OpenAIEmbedder(model="text-embedding-3-small")

# Local — free, runs on CPU/GPU
embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
embedder = SentenceTransformerEmbedder(model="BAAI/bge-large-en-v1.5")
```

### Retrievers

| Retriever | Dependencies | Use case |
|-----------|-------------|----------|
| `FaissRetriever` | faiss-cpu | High performance, persistence, production |
| `NumpyRetriever` | numpy only | Zero-dep, testing, small datasets |

```python
from ragpipe.retrievers import FaissRetriever, NumpyRetriever

# FAISS with disk persistence
retriever = FaissRetriever(dim=1536, persist_dir="./my_index")

# Pure NumPy — no extra install
retriever = NumpyRetriever()
```

### Rerankers

```python
from ragpipe.rerankers import CrossEncoderReranker

# Second-stage reranking with cross-encoder
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### Generators

```python
from ragpipe.generators import OpenAIGenerator

generator = OpenAIGenerator(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1024,
    system_prompt="Your custom system prompt...",
)
```

### Document Loaders

```python
from ragpipe.loaders import TextLoader, PDFLoader, DocxLoader, DirectoryLoader

# Single files
doc = TextLoader().load("notes.md")
doc = PDFLoader().load("paper.pdf")
doc = DocxLoader().load("report.docx")

# Entire directory (recursive, auto-detects format)
docs = DirectoryLoader().load("./documents/")
```

## Evaluation

Built-in retrieval quality metrics — no external eval framework needed.

```python
from ragpipe.evaluation import hit_rate, mrr, precision_at_k, recall_at_k, ndcg_at_k

results = pipe.retrieve("What is X?", top_k=10)
relevant = {"doc_id_1", "doc_id_2"}  # ground truth

print(f"Hit Rate:  {hit_rate(results, relevant)}")
print(f"MRR:       {mrr(results, relevant):.3f}")
print(f"P@5:       {precision_at_k(results, relevant, k=5):.3f}")
print(f"R@5:       {recall_at_k(results, relevant, k=5):.3f}")
print(f"NDCG@5:    {ndcg_at_k(results, relevant, k=5):.3f}")
```

Faithfulness scoring (n-gram overlap heuristic):

```python
from ragpipe.evaluation import faithfulness_score

scores = faithfulness_score(
    answer=result.answer,
    source_texts=[s.chunk.text for s in result.sources],
)
print(f"Unigram overlap: {scores['unigram_overlap']:.2%}")
print(f"Bigram overlap:  {scores['bigram_overlap']:.2%}")
```

## Custom Components

Every component is a base class you can extend:

```python
from ragpipe.embedders.base import BaseEmbedder

class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self._dim = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Your Ollama embedding logic here
        ...

    @property
    def dim(self) -> int:
        return self._dim
```

Works with any provider: Ollama, Cohere, Voyage AI, HuggingFace Inference, etc.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Design Decisions

- **FAISS IndexFlatIP with L2-normalization** — exact cosine similarity without approximate search overhead. At <100K vectors, brute-force is fast enough and eliminates index tuning complexity.
- **tiktoken cl100k_base encoding** — same tokenizer as GPT-4 and text-embedding-3, so token counts match what the API actually processes.
- **Separate retrieve() and query() methods** — retrieve() returns chunks without generation, essential for evaluation and debugging. query() runs the full pipeline.
- **Optional dependencies** — core package needs only numpy and tiktoken. FAISS, OpenAI, sentence-transformers are opt-in via extras.
- **No async in core** — sync-first API for simplicity. Wrap with asyncio if needed.

## License

MIT
