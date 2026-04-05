# ragpipe

Production-grade modular RAG framework with async-first architecture, hybrid search, streaming generation, REST API server, and pluggable components. Build advanced retrieval pipelines in 10 lines — or serve them as an API.

[![CI](https://github.com/shivamongit/ragpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/shivamongit/ragpipe/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-70%20passed-brightgreen.svg)]()

## Why ragpipe?

Naive RAG is dead. "Chunk → embed → cosine similarity → stuff into prompt" was always a prototype. Production RAG in 2026 needs async pipelines, hybrid search, streaming responses, and proper evaluation. ragpipe gives you all of this:

- **Async-first** — `aingest()`, `aquery()`, `aretrieve()`, `stream_query()` with native async in all providers
- **Streaming generation** — `stream()` / `astream()` on every generator, token-by-token WebSocket support
- **REST API server** — `python -m ragpipe serve` — FastAPI with `/ingest`, `/query`, `/query/stream` (WebSocket), `/stats`
- **Hybrid Search** — dense vectors (FAISS/NumPy/Chroma/Qdrant) + sparse keywords (BM25) with Reciprocal Rank Fusion
- **Contextual Chunking** — Anthropic's contextual retrieval approach (49% fewer retrieval failures)
- **Query Expansion** — HyDE, multi-query rewriting, step-back prompting
- **YAML pipeline config** — `Pipeline.from_yaml("config.yml")` for declarative setup
- **5 chunking strategies** — token, recursive, semantic, contextual, bring-your-own
- **6 retriever backends** — FAISS, NumPy, BM25, Hybrid (RRF), ChromaDB, Qdrant
- **6 embedder providers** — Ollama, Sentence Transformers, OpenAI, Voyage AI, Jina AI, bring-your-own
- **4 generator providers** — Ollama, OpenAI, Anthropic Claude, LiteLLM (100+ models)
- **Cross-encoder reranking** — second-stage precision refinement
- **9 evaluation metrics** — Hit Rate, MRR, P@K, R@K, NDCG@K, MAP@K, ROUGE-L, Context Precision, Faithfulness
- **Document loaders** — PDF, DOCX, TXT, Markdown, recursive directory
- **Zero cloud lock-in** — run entirely local with Ollama, or use any cloud provider, or mix both

## Install

```bash
pip install ragpipe                          # core (chunkers + numpy + BM25 + httpx)
pip install 'ragpipe[server]'               # + FastAPI REST API server
pip install 'ragpipe[openai]'               # + OpenAI embedder & generator
pip install 'ragpipe[anthropic]'            # + Anthropic Claude generator
pip install 'ragpipe[litellm]'              # + LiteLLM (100+ models)
pip install 'ragpipe[chroma]'               # + ChromaDB retriever
pip install 'ragpipe[qdrant]'               # + Qdrant retriever
pip install 'ragpipe[faiss]'                # + FAISS retriever
pip install 'ragpipe[config]'               # + YAML pipeline config
pip install 'ragpipe[all]'                  # everything
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
    ┌─────────────────────────────────────────────────────────────────────┐
    │               ragpipe Pipeline (sync + async + streaming)          │
    │  python -m ragpipe serve   ──▶   FastAPI + WebSocket server        │
    │  Pipeline.from_yaml(...)   ──▶   Declarative YAML config           │
    └─────────────────────────────────────────────────────────────────────┘
                                        │
     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
     │  Loader  │───▶│ Chunker  │───▶│ Embedder │───▶│Retriever │───▶│Reranker  │──▶ Generator
     └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
          │               │               │               │               │
     TextLoader      TokenChunker     Ollama         FAISS          CrossEncoder
     PDFLoader       Recursive        SentTrans      NumPy
     DocxLoader      Semantic         OpenAI         BM25                Ollama
     Directory       Contextual       Voyage AI      Hybrid (RRF)        OpenAI
                                      Jina AI        ChromaDB            Anthropic
                                                     Qdrant              LiteLLM (100+)
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
| `VoyageEmbedder` | No (API) | Paid | 512–1536 |
| `JinaEmbedder` | No (API) | Paid | 512–1024 |

```python
from ragpipe.embedders import OllamaEmbedder, JinaEmbedder

embedder = OllamaEmbedder(model="nomic-embed-text")              # local, free
embedder = JinaEmbedder(model="jina-embeddings-v3")               # 8192 context

# Optional (needs pip install):
from ragpipe.embedders import SentenceTransformerEmbedder, OpenAIEmbedder, VoyageEmbedder
embedder = SentenceTransformerEmbedder(model="BAAI/bge-large-en-v1.5")
embedder = OpenAIEmbedder(model="text-embedding-3-small")
embedder = VoyageEmbedder(model="voyage-3")
```

### Retrievers

| Retriever | Type | Dependencies | Best for |
|-----------|------|-------------|----------|
| `NumpyRetriever` | Dense | numpy | Zero-dep, testing |
| `FaissRetriever` | Dense | faiss-cpu | Production, persistence |
| `ChromaRetriever` | Dense | chromadb | Persistent local, metadata filtering |
| `QdrantRetriever` | Dense | qdrant-client | Scalable cloud + self-hosted |
| `BM25Retriever` | Sparse | none | Exact keyword matching |
| `HybridRetriever` | Dense + Sparse | none | **Best overall recall** |

```python
from ragpipe.retrievers import NumpyRetriever, BM25Retriever, HybridRetriever

retriever = NumpyRetriever()
retriever = BM25Retriever(k1=1.5, b=0.75)
retriever = HybridRetriever(
    dense_retriever=NumpyRetriever(),
    sparse_retriever=BM25Retriever(),
    dense_weight=0.6, sparse_weight=0.4,
)

# Optional (needs pip install):
from ragpipe.retrievers import FaissRetriever, ChromaRetriever, QdrantRetriever
retriever = FaissRetriever(dim=768, persist_dir="./index")
retriever = ChromaRetriever(collection_name="docs", persist_dir="./chroma_db")
retriever = QdrantRetriever(collection_name="docs", dim=768, url="http://localhost:6333")
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

| Generator | Runs locally | Cost | Models |
|-----------|-------------|------|--------|
| `OllamaGenerator` | Yes | Free | gemma4, qwen3.5, llama3.3, phi-4 |
| `OpenAIGenerator` | No (API) | Paid | gpt-4o-mini, gpt-4o |
| `AnthropicGenerator` | No (API) | Paid | claude-sonnet-4-20250514, claude-3.5-haiku |
| `LiteLLMGenerator` | Depends | Varies | 100+ models via single interface |

```python
from ragpipe.generators import OllamaGenerator

generator = OllamaGenerator(model="gemma4:26b", temperature=0.1)  # local, free

# Optional (needs pip install):
from ragpipe.generators import OpenAIGenerator, AnthropicGenerator, LiteLLMGenerator
generator = OpenAIGenerator(model="gpt-4o-mini")
generator = AnthropicGenerator(model="claude-sonnet-4-20250514")
generator = LiteLLMGenerator(model="gemini/gemini-2.0-flash")   # any of 100+ models
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

## Async-First Pipeline

Every method has a native async counterpart. Base classes default to `asyncio.to_thread` — all existing subclasses get async for free. Providers with HTTP I/O (Ollama, OpenAI, Anthropic) override with native async via `httpx.AsyncClient`.

```python
import asyncio
from ragpipe import Document, Pipeline

async def main():
    pipe = Pipeline(...)

    # Async ingest
    await pipe.aingest([Document(content="...")])

    # Async query
    result = await pipe.aquery("What are the key findings?")

    # Async retrieve (without generation)
    chunks = await pipe.aretrieve("What is FAISS?", top_k=10)

    # Streaming query — yields tokens as they arrive
    async for token in pipe.stream_query("Explain hybrid search"):
        print(token, end="", flush=True)

asyncio.run(main())
```

All generators support `stream()` (sync) and `astream()` (async) for token-by-token output.

## REST API Server

Serve any pipeline as an API with one command:

```bash
pip install 'ragpipe[server]'
python -m ragpipe serve --config pipeline.yml --port 8000 --api-key mysecret
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Ingest documents (JSON body) |
| `POST` | `/query` | Full RAG query → JSON response |
| `WS` | `/query/stream` | WebSocket streaming query |
| `GET` | `/stats` | Document and chunk counts |
| `DELETE` | `/index` | Clear the index |
| `POST` | `/evaluate` | Run eval metrics on a question |
| `GET` | `/health` | Health check |

```python
import httpx

r = httpx.post("http://localhost:8000/query", json={"question": "What is FAISS?"})
print(r.json()["answer"])
```

## YAML Pipeline Config

Define pipelines declaratively — no Python needed for deployment:

```yaml
# pipeline.yml
chunker:
  type: recursive
  chunk_size: 512
  overlap: 64
embedder:
  type: ollama
  model: nomic-embed-text
retriever:
  type: hybrid
  dense: {type: numpy}
  sparse: {type: bm25}
generator:
  type: ollama
  model: gemma4
top_k: 5
```

```python
from ragpipe.config import PipelineConfig

pipe = PipelineConfig.from_yaml("pipeline.yml").build()

# Or from a dict
pipe = PipelineConfig.from_dict({...}).build()

# Serialize current pipeline
config.to_yaml("pipeline.yml")
```

## Custom Components

Every component is a base class you can extend:

```python
from ragpipe.embedders.base import BaseEmbedder

class CohereEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def aembed(self, texts: list[str]) -> list[list[float]]: ...  # optional async override
    @property
    def dim(self) -> int: ...
```

Works with any provider. Custom components plug directly into `Pipeline`, the REST API server, and YAML config (via the component registry).

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v     # 70 tests
```

## Design Decisions

- **Async-first** — all base classes provide `asyncio.to_thread` defaults. Native async in HTTP providers (Ollama, OpenAI, Anthropic). Full backward compatibility.
- **Streaming** — every generator supports `stream()` / `astream()`. WebSocket streaming in the server.
- **Hybrid by default** — BM25 catches exact keyword matches that embeddings miss. RRF fusion requires no score normalization.
- **6 vector stores** — from zero-dep NumPy to production ChromaDB/Qdrant. Pick what fits your scale.
- **4 generator providers** — Ollama (free), OpenAI, Anthropic, LiteLLM (100+ models). All with streaming.
- **YAML config** — declarative pipeline definition for deployment. Component registry maps type strings to classes.
- **Contextual chunking** — prepending document-level context to chunks (Anthropic's approach) reduces retrieval failures by 49%.
- **Query expansion** — HyDE searches by hypothetical answer. Multi-query covers terminology gaps.
- **Ollama-first** — run the entire pipeline locally for $0/month.
- **Optional deps** — core needs only numpy + tiktoken + httpx. Everything else is opt-in.

## License

MIT
