<div align="center">

# ragpipe

### Production-Grade RAG Framework for Python

Build, evaluate, and deploy retrieval-augmented generation pipelines with enterprise features — agentic routing, semantic caching, conversation memory, observability, and 131 tests.

[![CI](https://github.com/shivamongit/ragpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/shivamongit/ragpipe/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-131%20passed-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-2.1.0-orange.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

[Quickstart](#quickstart) &#8226; [Features](#features) &#8226; [Install](#install) &#8226; [Architecture](#architecture) &#8226; [Docs](#components) &#8226; [Roadmap](ROADMAP.md)

</div>

---

## Why ragpipe?

Naive RAG is dead. Production RAG in 2026 needs **agentic query routing**, **semantic caching**, **conversation memory**, **LLM-as-Judge evaluation**, and **pipeline observability** — on top of async pipelines, hybrid search, and streaming responses.

ragpipe is a single framework that gives you all of this.

## Features

<table>
<tr>
<td width="50%">

**Core Pipeline**
- 6 chunking strategies (token, recursive, semantic, contextual, parent-child, custom)
- 6 embedder providers (Ollama, SentenceTransformers, OpenAI, Voyage, Jina, custom)
- 6 retriever backends (FAISS, NumPy, BM25, Hybrid RRF, ChromaDB, Qdrant)
- 4 generator providers (Ollama, OpenAI, Anthropic, LiteLLM 100+)
- Cross-encoder reranking
- Query expansion (HyDE, Multi-Query, Step-Back)

</td>
<td width="50%">

**Intelligence Layer** *(v2.1)*
- Agentic RAG Router (direct / single / multi-step / summarize)
- Parent-Child Chunking (small-to-big retrieval)
- Semantic Query Cache + Embedding Cache
- LLM-as-Judge Evaluation (faithfulness, relevance, completeness)
- Conversation Memory (multi-turn RAG)
- Pipeline Observability & Tracing

</td>
</tr>
<tr>
<td width="50%">

**Production Infrastructure**
- Async-first (`aingest`, `aquery`, `aretrieve`, `stream_query`)
- Streaming generation on every provider
- FastAPI REST API + WebSocket streaming
- YAML declarative pipeline config
- 10+ document loaders (PDF, DOCX, TXT, CSV, HTML, YouTube, directory)

</td>
<td width="50%">

**Developer Experience**
- 131 tests, 0.5s total runtime
- Core needs only `numpy` + `rs-bpe` + `httpx`
- Everything else is opt-in via extras
- Every component is an extensible base class
- April 2026 models (GPT-5.4, Claude 4.6, Gemini 3.1)
- Zero cloud lock-in — runs fully local with Ollama

</td>
</tr>
</table>

---

## Install

```bash
pip install ragpipe                          # core (chunkers + retrieval + BM25)
```

<details>
<summary><b>Optional extras</b></summary>

```bash
pip install 'ragpipe[server]'               # + FastAPI REST API server
pip install 'ragpipe[openai]'               # + OpenAI embedder & generator
pip install 'ragpipe[anthropic]'            # + Anthropic Claude generator
pip install 'ragpipe[litellm]'              # + LiteLLM (100+ models)
pip install 'ragpipe[faiss]'                # + FAISS retriever
pip install 'ragpipe[chroma]'               # + ChromaDB retriever
pip install 'ragpipe[qdrant]'               # + Qdrant retriever
pip install 'ragpipe[config]'               # + YAML pipeline config
pip install 'ragpipe[data]'                 # + CSV/Excel loaders (pandas)
pip install 'ragpipe[web]'                  # + HTML/Web loaders (BeautifulSoup)
pip install 'ragpipe[youtube]'              # + YouTube transcript loader
pip install 'ragpipe[all]'                  # everything
```

</details>

---

## Quickstart

### 10-Line Hybrid Search Pipeline

```python
from ragpipe import Document, Pipeline
from ragpipe.chunkers import RecursiveChunker
from ragpipe.embedders import OllamaEmbedder
from ragpipe.retrievers import NumpyRetriever, BM25Retriever, HybridRetriever
from ragpipe.generators import OllamaGenerator

pipe = Pipeline(
    chunker=RecursiveChunker(chunk_size=512, overlap=64),
    embedder=OllamaEmbedder(model="nomic-embed-text"),
    retriever=HybridRetriever(
        dense_retriever=NumpyRetriever(),
        sparse_retriever=BM25Retriever(),
    ),
    generator=OllamaGenerator(model="gemma4"),
)

pipe.ingest([Document(content="Your document text...", metadata={"source": "report.pdf"})])
result = pipe.query("What are the key findings?")
print(result.answer)
```

### Agentic RAG Router

```python
from ragpipe.agents import QueryRouter

router = QueryRouter(
    classify_fn=my_llm,           # LLM classifies query complexity
    retrieval_fn=my_retrieve,     # your retrieval function
    generation_fn=my_generate,    # your generation function
)

# Automatically routes: direct answer / single retrieval / multi-step / summarize
result = router.query("Compare the Q1 and Q2 financial results")
```

### Conversation Memory

```python
from ragpipe.memory import ConversationMemory

memory = ConversationMemory(
    contextualize_fn=my_llm,      # rewrites follow-ups as standalone queries
    max_history=20,
)

# Multi-turn: automatically contextualizes follow-up questions
result = memory.query("What were the key findings?", retrieval_fn=my_pipeline)
result = memory.query("How does that compare to last quarter?", retrieval_fn=my_pipeline)
```

### Semantic Caching

```python
from ragpipe.cache import SemanticCache

cache = SemanticCache(embed_fn=my_embed, threshold=0.95, ttl=3600)

# Skips retrieval + generation if a similar query was recently answered
cached = cache.get("What is FAISS?")
if cached is None:
    result = pipe.query("What is FAISS?")
    cache.put("What is FAISS?", result.answer, embedding)
```

### LLM-as-Judge Evaluation

```python
from ragpipe.evaluation import LLMJudge

judge = LLMJudge(judge_fn=my_llm)
scores = judge.evaluate(
    question="What is RAG?",
    answer=result.answer,
    context=[s.chunk.text for s in result.sources],
)
print(scores)  # {"faithfulness": 4.5, "relevance": 5.0, "completeness": 4.0, "overall": 4.5}
```

### Pipeline Observability

```python
from ragpipe.observability import Tracer

tracer = Tracer()
with tracer.span("retrieval", metadata={"top_k": 5}):
    results = pipe.retrieve("What is X?")
with tracer.span("generation"):
    answer = pipe.query("What is X?")

print(tracer.summary())    # per-step timing breakdown
print(tracer.to_json())    # structured JSON trace for logging
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ragpipe v2.1 — Intelligent RAG Framework                     │
│                                                                                 │
│  python -m ragpipe serve     ──▶   FastAPI + WebSocket server                   │
│  Pipeline.from_yaml(...)     ──▶   Declarative YAML config                      │
│  QueryRouter.query(...)      ──▶   Agentic routing (direct/single/multi/sum)    │
│  ConversationMemory.query()  ──▶   Multi-turn with auto-contextualization       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  Loader  │─▶│ Chunker  │─▶│ Embedder │─▶│Retriever │─▶│ Reranker │─▶│Generator │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
       │             │             │             │                            │
  Text/PDF/     Token         Ollama        FAISS          Cross-         Ollama
  DOCX/CSV/     Recursive     SentTrans     NumPy          Encoder        OpenAI
  HTML/Web      Semantic      OpenAI        BM25                          Anthropic
  YouTube       Contextual    Voyage        Hybrid RRF                    LiteLLM
  Directory     ParentChild   Jina          ChromaDB                      (100+)
                                            Qdrant
       │                                                        │
  ┌────┴────┐   ┌────────────┐   ┌───────────┐   ┌─────────────┴─────────────┐
  │ Semantic│   │  LLM-as-   │   │  Pipeline  │   │     Query Expansion       │
  │  Cache  │   │   Judge    │   │   Tracer   │   │  HyDE / Multi / StepBack  │
  └─────────┘   └────────────┘   └────────────┘   └───────────────────────────┘
```

---

## Components

### Chunkers

| Chunker | Strategy | Best for |
|---------|----------|----------|
| `TokenChunker` | Fixed token window + overlap | General purpose, predictable sizes |
| `RecursiveChunker` | Hierarchical separators (paragraph → sentence → word) | Preserving document structure |
| `SemanticChunker` | Split at embedding similarity breakpoints | Maximum coherence per chunk |
| `ContextualChunker` | Prepend LLM-generated document context | 49% fewer retrieval failures (Anthropic) |
| `ParentChildChunker` | Index small children, return large parent context | **Best precision + context trade-off** |

### Embedders

| Embedder | Local | Cost | Dimensions |
|----------|:-----:|------|:----------:|
| `OllamaEmbedder` | Yes | Free | 384–1024 |
| `SentenceTransformerEmbedder` | Yes | Free | 384–1024 |
| `OpenAIEmbedder` | No | Paid | 1536–3072 |
| `VoyageEmbedder` | No | Paid | 512–1536 |
| `JinaEmbedder` | No | Paid | 512–1024 |

### Retrievers

| Retriever | Type | Dependencies | Best for |
|-----------|------|:-------------|----------|
| `NumpyRetriever` | Dense | numpy | Zero-dep, testing |
| `FaissRetriever` | Dense | faiss-cpu | Production, persistence |
| `ChromaRetriever` | Dense | chromadb | Persistent local, metadata filtering |
| `QdrantRetriever` | Dense | qdrant-client | Scalable cloud + self-hosted |
| `BM25Retriever` | Sparse | none | Exact keyword matching |
| `HybridRetriever` | Dense + Sparse | none | **Best overall recall** |

### Generators

| Generator | Local | Cost | Top models (April 2026) |
|-----------|:-----:|------|------------------------|
| `OllamaGenerator` | Yes | Free | gemma4, qwen3.5, llama4:scout, deepseek-v3.2 |
| `OpenAIGenerator` | No | Paid | gpt-5.4, gpt-5.4-pro, gpt-5.3-codex, gpt-5-mini |
| `AnthropicGenerator` | No | Paid | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |
| `LiteLLMGenerator` | Mixed | Varies | All above + gemini-3.1-pro, mistral-large-3 |

### Document Loaders

| Loader | Formats | Dependencies |
|--------|---------|:-------------|
| `TextLoader` | `.txt`, `.md` | none |
| `PDFLoader` | `.pdf` | pypdf2 |
| `DocxLoader` | `.docx` | python-docx |
| `CSVLoader` | `.csv`, `.xlsx` | pandas |
| `HTMLLoader` | `.html`, URLs | beautifulsoup4 |
| `YouTubeLoader` | YouTube URLs | youtube-transcript-api |
| `DirectoryLoader` | recursive scan | auto-detects |

---

## Evaluation

### 9 Retrieval & Generation Metrics

```python
from ragpipe.evaluation import (
    hit_rate, mrr, precision_at_k, recall_at_k, ndcg_at_k, map_at_k,
    rouge_l, context_precision, faithfulness_score,
)
```

### LLM-as-Judge

Automated scoring of RAG output quality across three dimensions:

| Dimension | What it measures |
|-----------|-----------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Relevance** | Does the answer address the question? |
| **Completeness** | Does the answer cover all key points from the context? |

---

## Async & Streaming

Every method has a native async counterpart. Providers with HTTP I/O override with native async via `httpx.AsyncClient`.

```python
import asyncio
from ragpipe import Document, Pipeline

async def main():
    pipe = Pipeline(...)
    await pipe.aingest([Document(content="...")])
    result = await pipe.aquery("What are the key findings?")

    # Streaming — yields tokens as they arrive
    async for token in pipe.stream_query("Explain hybrid search"):
        print(token, end="", flush=True)

asyncio.run(main())
```

---

## REST API Server

```bash
python -m ragpipe serve --config pipeline.yml --port 8000 --api-key mysecret
```

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Ingest documents |
| `POST` | `/query` | RAG query → JSON |
| `WS` | `/query/stream` | WebSocket streaming |
| `GET` | `/stats` | Document & chunk counts |
| `DELETE` | `/index` | Clear index |
| `POST` | `/evaluate` | Run eval metrics |
| `GET` | `/health` | Health check |

---

## YAML Pipeline Config

```yaml
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
```

---

## Extending ragpipe

Every component is a base class. Custom implementations plug into Pipeline, the REST server, and YAML config automatically.

```python
from ragpipe.embedders.base import BaseEmbedder

class CohereEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def aembed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dim(self) -> int: ...
```

---

## Testing

```bash
pip install -e ".[dev]"
python run_tests.py          # 131 tests with per-test timing
```

---

## Project Stats

| Metric | Value |
|--------|-------|
| **Version** | 2.1.0 |
| **Tests** | 131 passed |
| **Test time** | < 1 second |
| **Core deps** | numpy, rs-bpe, httpx |
| **Python** | 3.10+ |
| **License** | MIT |

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full transformation plan.

| Phase | Status | Highlights |
|-------|--------|------------|
| **Phase 1** — Production Foundation | ✅ Complete | Async, streaming, REST API, 6 vector stores, YAML config |
| **Phase 2** — Intelligent Retrieval | ✅ Complete | Agentic router, parent-child chunking, caching, memory, observability |
| **Phase 3** — Intelligence Layer | Planned | Graph RAG, guardrails, PII protection, Gradio UI |

---

## License

MIT
