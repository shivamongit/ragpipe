# ragpipe

Production-grade modular RAG framework with hybrid search, contextual chunking, query expansion, and pluggable components. Build advanced retrieval pipelines in 10 lines.

[![CI](https://github.com/shivamongit/ragpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/shivamongit/ragpipe/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-54%20passed-brightgreen.svg)]()

## Why ragpipe?

Naive RAG is dead. "Chunk → embed → cosine similarity → stuff into prompt" was always a prototype. Production RAG in 2026 needs hybrid search, contextual retrieval, query expansion, and proper evaluation. ragpipe gives you all of this:

- **Hybrid Search** — combine dense vectors (FAISS/NumPy) + sparse keywords (BM25) with Reciprocal Rank Fusion
- **Contextual Chunking** — Anthropic's contextual retrieval approach (49% fewer retrieval failures)
- **Query Expansion** — HyDE, multi-query rewriting, step-back prompting
- **5 chunking strategies** — token, recursive, semantic, contextual, bring-your-own
- **4 retriever backends** — FAISS, NumPy, BM25, Hybrid (RRF)
- **3 embedder providers** — OpenAI, Sentence Transformers, Ollama (local, free)
- **2 generator providers** — OpenAI, Ollama (local, free)
- **Cross-encoder reranking** — second-stage precision refinement
- **9 evaluation metrics** — Hit Rate, MRR, P@K, R@K, NDCG@K, MAP@K, ROUGE-L, Context Precision, Faithfulness
- **Document loaders** — PDF, DOCX, TXT, Markdown, recursive directory
- **Zero cloud lock-in** — run entirely local with Ollama, or use OpenAI, or mix both

## Install

```bash
pip install ragpipe                          # core (chunkers + numpy + BM25)
pip install 'ragpipe[faiss]'                 # + FAISS retriever
pip install 'ragpipe[openai]'                # + OpenAI embedder & generator
pip install 'ragpipe[sentence-transformers]' # + local embedder & reranker
pip install 'ragpipe[all]'                   # everything
```

## Quickstart — Hybrid Search Pipeline

```python
from ragpipe import Document, Pipeline
from ragpipe.chunkers import RecursiveChunker
from ragpipe.embedders import OllamaEmbedder          # local, free
from ragpipe.retrievers import NumpyRetriever, BM25Retriever, HybridRetriever
from ragpipe.rerankers import CrossEncoderReranker
from ragpipe.generators import OllamaGenerator         # local, free

embedder = OllamaEmbedder(model="nomic-embed-text")

pipe = Pipeline(
    chunker=RecursiveChunker(chunk_size=512, overlap=64),
    embedder=embedder,
    retriever=HybridRetriever(
        dense_retriever=NumpyRetriever(),
        sparse_retriever=BM25Retriever(),
    ),
    generator=OllamaGenerator(model="gemma4"),
    top_k=5,
)

pipe.ingest([Document(content="Your document text...", metadata={"source": "report.pdf"})])

result = pipe.query("What are the key findings?")
print(result.answer)         # cited answer
print(result.sources)        # ranked source chunks with scores
print(result.latency_ms)     # end-to-end latency
```

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              ragpipe Pipeline                │
                    └─────────────────────────────────────────────┘
                                        │
     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
     │  Loader  │───▶│ Chunker  │───▶│ Embedder │───▶│Retriever │───▶│Reranker  │──▶ Generator
     └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
          │               │               │               │               │
     TextLoader      TokenChunker     OpenAI         FAISS          CrossEncoder
     PDFLoader       Recursive        SentTrans      NumPy
     DocxLoader      Semantic         Ollama         BM25
     Directory       Contextual                      Hybrid (RRF)
                                                        │
                                           ┌────────────┴────────────┐
                                           │  Query Expansion        │
                                           │  • HyDE                 │
                                           │  • Multi-Query          │
                                           │  • Step-Back Prompting  │
                                           └─────────────────────────┘
```

## Components

### Chunkers

| Chunker | Strategy | Best for |
|---------|----------|----------|
| `TokenChunker` | Fixed token window + overlap | General purpose, predictable sizes |
| `RecursiveChunker` | Hierarchical separators (¶ → sentence → word) | Preserving document structure |
| `SemanticChunker` | Split at embedding similarity breakpoints | Maximum coherence per chunk |
| `ContextualChunker` | Prepend LLM-generated document context | **49% fewer retrieval failures** (Anthropic) |

```python
from ragpipe.chunkers import TokenChunker, RecursiveChunker, SemanticChunker, ContextualChunker

chunker = TokenChunker(chunk_size=512, overlap=64)
chunker = RecursiveChunker(chunk_size=512, overlap=64)
chunker = SemanticChunker(embedder=my_embedder, threshold=0.75)

# Contextual: wraps any base chunker + LLM to add document context
chunker = ContextualChunker(
    base_chunker=RecursiveChunker(chunk_size=512),
    context_generator=my_llm_call,  # callable(prompt) -> str
)
```

### Embedders

| Embedder | Runs locally | Cost | Dimensions |
|----------|-------------|------|------------|
| `OllamaEmbedder` | Yes | Free | 384–1024 |
| `SentenceTransformerEmbedder` | Yes | Free | 384–1024 |
| `OpenAIEmbedder` | No (API) | Paid | 1536–3072 |

```python
from ragpipe.embedders import OllamaEmbedder, SentenceTransformerEmbedder, OpenAIEmbedder

embedder = OllamaEmbedder(model="nomic-embed-text")              # local, free
embedder = SentenceTransformerEmbedder(model="BAAI/bge-large-en-v1.5")  # local
embedder = OpenAIEmbedder(model="text-embedding-3-small")         # cloud
```

### Retrievers

| Retriever | Type | Dependencies | Best for |
|-----------|------|-------------|----------|
| `FaissRetriever` | Dense | faiss-cpu | Production, persistence |
| `NumpyRetriever` | Dense | numpy | Zero-dep, testing |
| `BM25Retriever` | Sparse | none | Exact keyword matching |
| `HybridRetriever` | Dense + Sparse | none | **Best overall recall** |

```python
from ragpipe.retrievers import FaissRetriever, NumpyRetriever, BM25Retriever, HybridRetriever

# Dense vector search
retriever = FaissRetriever(dim=768, persist_dir="./index")
retriever = NumpyRetriever()

# Sparse keyword search
retriever = BM25Retriever(k1=1.5, b=0.75)

# Hybrid: dense + sparse fused with Reciprocal Rank Fusion
retriever = HybridRetriever(
    dense_retriever=NumpyRetriever(),
    sparse_retriever=BM25Retriever(),
    dense_weight=0.6,       # favor semantic for most use cases
    sparse_weight=0.4,
)
```

### Query Expansion

Transform the raw user query before retrieval for better recall:

```python
from ragpipe.query import HyDEExpander, MultiQueryExpander, StepBackExpander

# HyDE: generate hypothetical answer, search for similar docs
expander = HyDEExpander(generate_fn=my_llm)
queries = expander.expand("What causes high API latency?")
# → ["What causes high API latency?", "High API latency is typically caused by..."]

# Multi-Query: generate diverse reformulations
expander = MultiQueryExpander(generate_fn=my_llm, n_queries=3)
queries = expander.expand("What causes high API latency?")
# → ["What causes high API latency?", "API performance bottlenecks", "slow response time debugging"]

# Step-Back: ask a broader question first
expander = StepBackExpander(generate_fn=my_llm)
queries = expander.expand("Why is our /api/users endpoint slow?")
# → ["Why is our /api/users endpoint slow?", "What are common causes of API performance issues?"]
```

### Generators

| Generator | Runs locally | Cost |
|-----------|-------------|------|
| `OllamaGenerator` | Yes | Free |
| `OpenAIGenerator` | No (API) | Paid |

```python
from ragpipe.generators import OllamaGenerator, OpenAIGenerator

generator = OllamaGenerator(model="gemma4:26b", temperature=0.1)  # local
generator = OpenAIGenerator(model="gpt-4o-mini", temperature=0.1)  # cloud
```

### Rerankers

```python
from ragpipe.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### Document Loaders

```python
from ragpipe.loaders import TextLoader, PDFLoader, DocxLoader, DirectoryLoader

doc = TextLoader().load("notes.md")
doc = PDFLoader().load("paper.pdf")
doc = DocxLoader().load("report.docx")
docs = DirectoryLoader().load("./documents/")  # recursive, auto-detects format
```

## Evaluation

9 built-in metrics — no external eval framework needed.

```python
from ragpipe.evaluation import (
    hit_rate, mrr, precision_at_k, recall_at_k, ndcg_at_k, map_at_k,
    rouge_l, context_precision, faithfulness_score,
)

results = pipe.retrieve("What is X?", top_k=10)
relevant = {"doc_id_1", "doc_id_2"}

# Retrieval quality
print(f"Hit Rate:       {hit_rate(results, relevant)}")
print(f"MRR:            {mrr(results, relevant):.3f}")
print(f"P@5:            {precision_at_k(results, relevant, k=5):.3f}")
print(f"R@5:            {recall_at_k(results, relevant, k=5):.3f}")
print(f"NDCG@5:         {ndcg_at_k(results, relevant, k=5):.3f}")
print(f"MAP@5:          {map_at_k(results, relevant, k=5):.3f}")
print(f"Context Prec:   {context_precision(results, relevant):.3f}")

# Generation quality
scores = rouge_l(answer=result.answer, reference="expected answer text")
print(f"ROUGE-L F1:     {scores['f1']:.3f}")

faith = faithfulness_score(result.answer, [s.chunk.text for s in result.sources])
print(f"Faithfulness:   {faith['unigram_overlap']:.2%}")
```

## Custom Components

Every component is a base class you can extend:

```python
from ragpipe.embedders.base import BaseEmbedder

class CohereEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dim(self) -> int: ...
```

Works with any provider: Ollama, Cohere, Voyage AI, HuggingFace Inference, etc.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v     # 54 tests
```

## Design Decisions

- **Hybrid by default** — BM25 catches exact keyword matches that embeddings miss. RRF fusion requires no score normalization.
- **Contextual chunking** — prepending document-level context to chunks (Anthropic's approach) reduces retrieval failures by 49%.
- **Query expansion** — HyDE searches by hypothetical answer (matches document language better than questions). Multi-query covers different phrasings.
- **FAISS IndexFlatIP with L2-normalization** — exact cosine similarity. Brute-force is fast enough under 100K vectors.
- **tiktoken cl100k_base** — same tokenizer as GPT-4 and text-embedding-3.
- **Ollama-first** — run the entire pipeline locally for $0/month with `OllamaEmbedder` + `OllamaGenerator`.
- **Optional deps** — core needs only numpy + tiktoken. Everything else is opt-in.

## License

MIT
