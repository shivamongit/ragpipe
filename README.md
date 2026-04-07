<div align="center">

<img src="https://img.shields.io/badge/🔬-ragpipe-blue?style=for-the-badge&labelColor=1a1a2e&color=16213e" alt="ragpipe" height="60">

# ragpipe

### The Intelligent RAG Framework

Build, evaluate, and deploy production-grade retrieval-augmented generation pipelines with **Knowledge Graph RAG**, **self-correcting CRAG**, **SelfRAG**, **ReAct agents**, **adaptive retrieval**, **hallucination detection**, **pipeline auto-tuning**, **guardrails**, and a composable **SmartPipeline** — all in one framework.

[![CI](https://github.com/shivamongit/ragpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/shivamongit/ragpipe/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-314%20passed-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-3.0.0-orange.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

[Quickstart](#-quickstart) &#8226; [Features](#-features) &#8226; [Install](#-install) &#8226; [Architecture](#-architecture) &#8226; [Docs](#-components) &#8226; [Roadmap](ROADMAP.md) &#8226; [Analysis](PROGRESS_REPORT.md)

</div>

---

## ⚡ Why ragpipe?

> **Naive RAG is dead.** Production RAG in 2026 needs knowledge graphs, self-correcting retrieval, adaptive strategy selection, reasoning agents, hallucination detection, PII guardrails, and automatic pipeline optimization — on top of agentic routing, caching, memory, and streaming.

**ragpipe is the only framework that gives you all of this in one place.** These features don't exist as built-in modules in LangChain, LlamaIndex, Haystack, or DSPy:

| Capability | ragpipe | LangChain | LlamaIndex | Haystack | DSPy |
|:-----------|:-------:|:---------:|:----------:|:--------:|:----:|
| Knowledge Graph RAG | ✅ Built-in | ❌ | Partial | ❌ | ❌ |
| Self-Correcting CRAG | ✅ Built-in | ❌ | ❌ | ❌ | ❌ |
| SelfRAG (reflection tokens) | ✅ Built-in | ❌ | ❌ | ❌ | ❌ |
| ReAct Agent (tool use) | ✅ Built-in | Partial | ❌ | ❌ | ❌ |
| Adaptive Retrieval | ✅ Built-in | ❌ | ❌ | ❌ | ❌ |
| Pipeline Auto-Tuning | ✅ Built-in | ❌ | ❌ | ❌ | Prompts only |
| Answer Verification | ✅ Built-in | ❌ | ❌ | ❌ | ❌ |
| Composable SmartPipeline | ✅ Built-in | ❌ | ❌ | ❌ | ❌ |
| Zero-Dep Guardrails | ✅ Built-in | External | ❌ | ❌ | ❌ |
| Community Detection | ✅ Built-in | ❌ | ❌ | ❌ | ❌ |

---

## 🚀 Features

<table>
<tr>
<td width="50%">

### Core Pipeline
- 🧩 **6 chunking strategies** — token, recursive, semantic, contextual, parent-child
- 🔍 **6 embedder providers** — Ollama, SentenceTransformers, OpenAI, Voyage, Jina
- 📦 **6 retriever backends** — FAISS, NumPy, BM25, Hybrid RRF, ChromaDB, Qdrant
- 🤖 **4 generator providers** — Ollama, OpenAI, Anthropic, LiteLLM (100+ models)
- ↔️ Cross-encoder reranking
- 🔎 Query expansion (HyDE, Multi-Query, Step-Back)
- 📄 10+ document loaders (PDF, DOCX, CSV, HTML, YouTube)

</td>
<td width="50%">

### Intelligence Layer
- 🕸️ **Knowledge Graph RAG** — entity extraction, graph builder, community detection, graph+vector hybrid retrieval
- 🧠 **Self-Correcting CRAG** — document relevance grading → refine → web fallback
- 🪞 **SelfRAG** — self-reflective retrieval with relevance/support/usefulness tokens
- ⚡ **ReAct Agent** — reasoning + acting loop with pluggable tools
- 🎯 **Adaptive Retrieval** — auto strategy + confidence + fallback chain
- 🔀 **SmartPipeline** — composable orchestrator wiring all modules together
- 📊 Pipeline Optimizer (DSPy-inspired auto-tuning)

</td>
</tr>
<tr>
<td width="50%">

### Safety & Evaluation
- ✅ **Answer Verifier** — claim-level hallucination detection & grounding
- 🛡️ **Guardrails** — PII redaction, injection detection, topic filtering
- ⚖️ **LLM-as-Judge** — faithfulness, relevance, completeness (0–5 scale)
- 📈 **9 metrics** — Hit Rate, MRR, P@K, R@K, NDCG, MAP, ROUGE-L, Context Precision, Faithfulness

</td>
<td width="50%">

### Production Infrastructure
- ⚡ Async-first (`aingest`, `aquery`, `aretrieve`, `stream_query`)
- 🌊 Streaming generation on every provider
- 🌐 FastAPI REST API + WebSocket streaming
- 📋 YAML declarative pipeline config
- 💾 Semantic cache + embedding cache
- 💬 Conversation memory (multi-turn RAG)
- 📊 Structured tracing & observability
- 🔒 API key authentication

</td>
</tr>
</table>

### Developer Experience

- **314 tests**, 0.74s total runtime — every module tested
- Core needs only `numpy` + `rs-bpe` + `httpx` (~10 MB)
- Everything else is opt-in via extras
- Every component is an extensible base class
- April 2026 models (GPT-5.4, Claude 4.6, Gemini 3.1)
- **Zero cloud lock-in** — runs fully local with Ollama

---

## 📦 Install

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

## 🎯 Quickstart

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

### 🕸️ Knowledge Graph RAG *(v3.0)*

```python
from ragpipe.graph import EntityExtractor, GraphBuilder, CommunityDetector, GraphRetriever

# Build knowledge graph from documents
extractor = EntityExtractor(extract_fn=my_llm)
builder = GraphBuilder(extractor=extractor)
result = builder.build(documents)

graph = result.graph
print(f"Entities: {graph.entity_count}, Relationships: {graph.relationship_count}")

# Detect communities for global search
detector = CommunityDetector(summarize_fn=my_llm)
communities = detector.detect(graph)

# Graph-enhanced retrieval (local + global + vector hybrid)
retriever = GraphRetriever(
    graph=graph,
    communities=communities,
    vector_retrieve_fn=pipe.retrieve,
    generate_fn=my_llm,
    strategy="hybrid",     # "local", "global", or "hybrid"
    max_hops=2,
)
result = retriever.retrieve("How does Company X relate to Project Y?")
print(result.answer)
print(result.entities_used)     # entities traversed
print(result.communities_used)  # communities consulted
```

### 🪞 SelfRAG Agent *(v3.0)*

```python
from ragpipe.agents import SelfRAGAgent

agent = SelfRAGAgent(
    retrieve_fn=my_retrieve,
    generate_fn=my_generate,
    reflect_fn=my_llm,          # LLM generates reflection tokens
    max_iterations=3,
)

result = agent.query("What is quantum computing?")
print(result.answer)
print(result.reflection.retrieval_needed)  # RetrievalDecision.RETRIEVE
print(result.reflection.support_level)     # SupportLevel.FULLY_SUPPORTED
print(result.reflection.usefulness)        # 5 (1-5 scale)
print(result.confidence)                   # 0.92
```

### ⚡ ReAct Agent *(v3.0)*

```python
from ragpipe.agents import ReActAgent, Tool

agent = ReActAgent(
    reason_fn=my_llm,
    tools=[
        Tool(name="search", description="Search the knowledge base", fn=my_search),
        Tool(name="calculate", description="Perform calculations", fn=my_calc),
        Tool(name="lookup", description="Look up entity details", fn=my_lookup),
    ],
    max_steps=5,
)

result = agent.query("What is 15% of Company X's revenue from the latest report?")
print(result.answer)
print(result.steps)       # [ReActStep(thought=..., action=..., observation=...), ...]
print(result.tools_used)  # ["search", "calculate"]
```

### 🔗 SmartPipeline — Composable Intelligence *(v3.0)*

```python
from ragpipe.agents import SmartPipeline
from ragpipe.guardrails import PIIRedactor, PromptInjectionDetector, TopicGuardrail
from ragpipe.cache import SemanticCache
from ragpipe.memory import ConversationMemory
from ragpipe.verification import AnswerVerifier

# Wire everything together — single .query() orchestrates all modules
smart = SmartPipeline(
    pipeline=my_pipeline,
    guardrails=[PromptInjectionDetector(), TopicGuardrail(allowed_topics=["finance"])],
    pii_redactor=PIIRedactor(),
    cache=SemanticCache(embed_fn=embed, threshold=0.95),
    memory=ConversationMemory(contextualize_fn=llm),
    verifier=AnswerVerifier(verify_fn=llm),
    on_guardrail_fail="block",
)

# One call: guardrails → cache → memory → route → retrieve → verify → respond
result = smart.query("What are the tax implications of stock options?")
print(result.answer)
print(result.guardrail_checks)  # {"injection": False, "topic": True}
print(result.verification)      # {"confidence": 0.95, "hallucination_rate": 0.0}
print(result.cached)            # False
print(result.latency_ms)        # 245.3
```

### 🧠 Self-Correcting CRAG Agent

```python
from ragpipe.agents import CRAGAgent

agent = CRAGAgent(
    grade_fn=my_llm,          # grades each doc: CORRECT / AMBIGUOUS / INCORRECT
    retrieve_fn=my_retrieve,
    generate_fn=my_generate,
    web_search_fn=my_search,   # optional web fallback
)

result = agent.query("What caused the 2024 market correction?")
print(result.answer)           # grounded answer
print(result.action_taken)     # direct_generate / refined_generate / web_search / no_answer
print(result.confidence)       # 0.0–1.0
```

### 🎯 Adaptive Retrieval

```python
from ragpipe.agents import AdaptiveRetriever

retriever = AdaptiveRetriever(
    strategies={"dense": dense_fn, "sparse": sparse_fn, "hybrid": hybrid_fn},
    confidence_threshold=0.3,
)

# Auto: classify query → select strategy → adjust top_k → retry on low confidence
result = retriever.retrieve("Compare FAISS vs ChromaDB performance")
print(result.strategy_used)     # RetrievalStrategy.MULTI_PASS
print(result.query_complexity)  # QueryComplexity.COMPARATIVE
```

<details>
<summary><b>More examples: Router, Memory, Cache, Optimizer, Verifier, Guardrails, Evaluation, Tracing</b></summary>

### Agentic RAG Router

```python
from ragpipe.agents import QueryRouter

router = QueryRouter(
    classify_fn=my_llm,
    retrieval_fn=my_retrieve,
    generation_fn=my_generate,
)
# Routes: direct answer / single retrieval / multi-step / summarize
result = router.query("Compare the Q1 and Q2 financial results")
```

### Conversation Memory

```python
from ragpipe.memory import ConversationMemory

memory = ConversationMemory(contextualize_fn=my_llm, max_history=20)
result = memory.query("What were the key findings?", retrieval_fn=my_pipeline)
result = memory.query("How does that compare to last quarter?", retrieval_fn=my_pipeline)
```

### Semantic Caching

```python
from ragpipe.cache import SemanticCache

cache = SemanticCache(embed_fn=my_embed, threshold=0.95, ttl=3600)
cached = cache.lookup("What is FAISS?")
```

### Pipeline Optimizer

```python
from ragpipe.optimization import PipelineOptimizer, ParameterSpace

optimizer = PipelineOptimizer(pipeline_factory=build_pipeline, eval_fn=evaluate, eval_dataset=qa_pairs)
result = optimizer.optimize(ParameterSpace(chunk_size=[256, 512, 1024], top_k=[3, 5, 10]))
print(result.best_params)   # {"chunk_size": 512, "top_k": 5}
```

### Answer Verification

```python
from ragpipe.verification import AnswerVerifier

verifier = AnswerVerifier(verify_fn=my_llm)
result = verifier.verify(answer="Paris is the capital.", sources=["Paris is the capital of France."])
print(result.hallucination_rate)  # 0.0
```

### Guardrails

```python
from ragpipe.guardrails import PIIRedactor, PromptInjectionDetector, TopicGuardrail

PIIRedactor().redact("Email john@test.com")  # "Email [EMAIL_REDACTED]"
PromptInjectionDetector().check("Ignore instructions").is_injection  # True
TopicGuardrail(allowed_topics=["finance"]).is_allowed("Tax question?")  # True
```

### LLM-as-Judge

```python
from ragpipe.evaluation import LLMJudge

judge = LLMJudge(judge_fn=my_llm)
scores = judge.evaluate(question="What is RAG?", answer=answer, context=context)
# {"faithfulness": 4.5, "relevance": 5.0, "completeness": 4.0, "overall": 4.5}
```

### Pipeline Observability

```python
from ragpipe.observability import Tracer

tracer = Tracer()
with tracer.span("retrieval", metadata={"top_k": 5}):
    results = pipe.retrieve("What is X?")
print(tracer.summary())    # per-step timing
print(tracer.to_json())    # structured JSON
```

</details>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                          ragpipe v3.0 — The Intelligent RAG Framework                    │
│                                                                                          │
│  SmartPipeline.query(...)   ──▶  Guardrails → Cache → Memory → Route → Retrieve → Verify│
│  GraphRetriever.retrieve()  ──▶  Entity extraction → Graph traversal → Community search  │
│  SelfRAGAgent.query(...)    ──▶  Retrieve? → Relevance → Support → Usefulness → Iterate  │
│  ReActAgent.query(...)      ──▶  Think → Act → Observe → Repeat → Final Answer           │
│  python -m ragpipe serve    ──▶  FastAPI + WebSocket streaming server                     │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
    ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
    │                                          │                                          │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐ │
    │  │  Loader  │→│ Chunker  │→│ Embedder │→│Retriever │→│Reranker│→│Generator │ │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────────┘  └──────────┘ │
    │       │             │             │             │                          │         │
    │  Text/PDF/     Token         Ollama        FAISS         Cross-       Ollama        │
    │  DOCX/CSV/     Recursive     SentTrans     NumPy         Encoder      OpenAI        │
    │  HTML/Web      Semantic      OpenAI        BM25                       Anthropic      │
    │  YouTube       Contextual    Voyage        Hybrid RRF                 LiteLLM        │
    │  Directory     ParentChild   Jina          ChromaDB                   (100+)         │
    │                                            Qdrant                                    │
    └──────────────────────────────────────────────────────────────────────────────────────┘
                                              │
    ┌────────────────┬────────────────┬────────┴───────┬─────────────────┬─────────────────┐
    │  Knowledge     │  Semantic      │  LLM-as-       │  Pipeline       │  Query           │
    │  Graph         │  Cache         │  Judge          │  Tracer         │  Expansion       │
    │  ┌───────────┐ │  ┌───────────┐ │  ┌───────────┐ │  ┌───────────┐ │  ┌─────────────┐ │
    │  │ Entities  │ │  │ Query     │ │  │ Faith.    │ │  │ Spans     │ │  │ HyDE        │ │
    │  │ Relations │ │  │ Embedding │ │  │ Relevance │ │  │ Timing    │ │  │ Multi-Query │ │
    │  │ Community │ │  │ LRU       │ │  │ Complete  │ │  │ JSON      │ │  │ Step-Back   │ │
    │  └───────────┘ │  └───────────┘ │  └───────────┘ │  └───────────┘ │  └─────────────┘ │
    └────────────────┴────────────────┴────────────────┴─────────────────┴─────────────────┘
```

---

## 🧩 Components

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
| `OllamaEmbedder` | ✅ | Free | 384–1024 |
| `SentenceTransformerEmbedder` | ✅ | Free | 384–1024 |
| `OpenAIEmbedder` | ☁️ | Paid | 1536–3072 |
| `VoyageEmbedder` | ☁️ | Paid | 512–1536 |
| `JinaEmbedder` | ☁️ | Paid | 512–1024 |

### Retrievers

| Retriever | Type | Dependencies | Best for |
|-----------|------|:-------------|----------|
| `NumpyRetriever` | Dense | numpy | Zero-dep, testing |
| `FaissRetriever` | Dense | faiss-cpu | Production, persistence |
| `ChromaRetriever` | Dense | chromadb | Persistent local, metadata filtering |
| `QdrantRetriever` | Dense | qdrant-client | Scalable cloud + self-hosted |
| `BM25Retriever` | Sparse | none | Exact keyword matching |
| `HybridRetriever` | Dense + Sparse | none | **Best overall recall** |
| `GraphRetriever` | Graph + Vector | none | **Multi-hop reasoning** *(v3.0)* |

### Generators

| Generator | Local | Cost | Top models (April 2026) |
|-----------|:-----:|------|------------------------|
| `OllamaGenerator` | ✅ | Free | gemma4, qwen3.5, llama4:scout, deepseek-v3.2 |
| `OpenAIGenerator` | ☁️ | Paid | gpt-5.4, gpt-5.4-pro, gpt-5.3-codex, gpt-5-mini |
| `AnthropicGenerator` | ☁️ | Paid | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |
| `LiteLLMGenerator` | Mixed | Varies | All above + gemini-3.1-pro, mistral-large-3 |

### Agents & Intelligence

| Agent | What it does | Key Innovation |
|-------|-------------|----------------|
| `QueryRouter` | Routes queries to optimal strategy | 4 route types with sub-question decomposition |
| `CRAGAgent` | Self-correcting retrieval | Document grading + knowledge refinement + web fallback |
| `SelfRAGAgent` | Self-reflective generation | Reflection tokens (relevance, support, usefulness) |
| `ReActAgent` | Reasoning + tool use | Think → Act → Observe loop with pluggable tools |
| `AdaptiveRetriever` | Dynamic strategy selection | Query complexity → strategy → confidence → retry |
| `SmartPipeline` | Composable orchestrator | Guardrails → cache → memory → route → verify |
| `GraphRetriever` | Graph-enhanced retrieval | Entity traversal + community search + vector fusion |

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

## 📊 Evaluation

### 9 Retrieval & Generation Metrics

```python
from ragpipe.evaluation import (
    hit_rate, mrr, precision_at_k, recall_at_k, ndcg_at_k, map_at_k,
    rouge_l, context_precision, faithfulness_score,
)
```

### LLM-as-Judge

| Dimension | What it measures | Scale |
|-----------|-----------------|:-----:|
| **Faithfulness** | Is the answer grounded in the retrieved context? | 0–5 |
| **Relevance** | Does the answer address the question? | 0–5 |
| **Completeness** | Does the answer cover all key points? | 0–5 |

---

## ⚡ Async & Streaming

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

## 🌐 REST API Server

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

## 📋 YAML Pipeline Config

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

## 🔌 Extending ragpipe

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

## 🧪 Testing

```bash
pip install -e ".[dev]"
python run_tests.py          # 314 tests with per-test timing
pytest tests/ -v             # standard pytest
```

---

## 📈 Project Stats

| Metric | Value |
|--------|-------|
| **Version** | 3.0.0 |
| **Tests** | 314 passed |
| **Test time** | 0.74 seconds |
| **Source lines** | ~10,000 |
| **Core deps** | numpy, rs-bpe, httpx |
| **Python** | 3.10+ |
| **License** | MIT |

---

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan.

| Phase | Version | Status | Highlights |
|-------|---------|--------|------------|
| **Phase 1** — Production Foundation | v2.0 | ✅ Complete | Async, streaming, REST API, 6 vector stores, YAML config |
| **Phase 2** — Intelligent Retrieval | v2.1 | ✅ Complete | Agentic router, parent-child chunking, caching, memory, observability |
| **Phase 3** — Intelligence Layer | v2.2 | ✅ Complete | CRAG, adaptive retrieval, optimizer, verifier, guardrails |
| **Phase 4** — Knowledge & Agents | v3.0 | ✅ Complete | Knowledge Graph RAG, SelfRAG, ReAct, SmartPipeline, 314 tests |
| **Phase 5** — Platform | v3.1 | 🔜 Next | Gradio UI, Docker, Helm, benchmarks, plugin system |

---

## 📄 License

MIT
