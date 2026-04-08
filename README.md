<div align="center">

<!-- Hero Banner -->
<br>

```
                                      ╔═══════════════════════╗
                                      ║                       ║
     ██████╗  █████╗  ██████╗         ║   🧠 Intelligence     ║
     ██╔══██╗██╔══██╗██╔════╝         ║   🕸️ Knowledge Graph  ║
     ██████╔╝███████║██║  ███╗        ║   ⚡ Self-Correcting  ║
     ██╔══██╗██╔══██║██║   ██║        ║   🔄 Adaptive         ║
     ██║  ██║██║  ██║╚██████╔╝        ║   🛡️ Guardrails       ║
     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝        ║                       ║
     ██████╗ ██╗██████╗ ███████╗      ╚═══════════════════════╝
     ██╔══██╗██║██╔══██╗██╔════╝
     ██████╔╝██║██████╔╝█████╗        The Intelligent
     ██╔═══╝ ██║██╔═══╝ ██╔══╝        RAG Framework
     ██║     ██║██║     ███████╗
     ╚═╝     ╚═╝╚═╝     ╚══════╝      v3.0  ·  April 2026
```

<br>

### Build production-grade RAG that **thinks**, **self-corrects**, and **reasons** — not just retrieves.

<br>

Knowledge Graph RAG · Self-Correcting CRAG · SelfRAG Reflection · ReAct Agents · Adaptive Retrieval
Pipeline Auto-Tuning · Hallucination Detection · PII Guardrails · Composable SmartPipeline

<br>

[![CI](https://github.com/shivamongit/ragpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/shivamongit/ragpipe/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-314%20passed-brightgreen.svg?style=flat-square)]()
[![Version](https://img.shields.io/badge/version-3.0.0-orange.svg?style=flat-square)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?style=flat-square)](https://github.com/astral-sh/ruff)
[![Downloads](https://img.shields.io/badge/downloads-growing-blue.svg?style=flat-square)]()

<br>

**[🚀 Quickstart](#-quickstart)** · **[✨ Features](#-feature-overview)** · **[📦 Install](#-install)** · **[🏗️ Architecture](#%EF%B8%8F-architecture)** · **[📖 Docs](#-components-reference)** · **[🗺️ Roadmap](ROADMAP.md)**

<br>

</div>

---

<div align="center">

## 🔥 Why ragpipe?

</div>

> **Naive RAG is dead.** In 2026, production RAG demands knowledge graphs, self-correcting retrieval, reasoning agents, hallucination detection, and automatic optimization — not just vector search.

**ragpipe is the only open-source framework that ships all of these as built-in, tested modules.**

<table>
<tr><td>

### 🏆 ragpipe vs. The Industry

</td></tr>
<tr><td>

| Capability | ragpipe | LangChain | LlamaIndex | Haystack | DSPy |
|:-----------|:-------:|:---------:|:----------:|:--------:|:----:|
| 🕸️ Knowledge Graph RAG | ✅ **Built-in** | ❌ | Partial | ❌ | ❌ |
| 🧠 Self-Correcting CRAG | ✅ **Built-in** | ❌ | ❌ | ❌ | ❌ |
| 🪞 SelfRAG (reflection tokens) | ✅ **Built-in** | ❌ | ❌ | ❌ | ❌ |
| ⚡ ReAct Agent (tool use) | ✅ **Built-in** | Partial | ❌ | ❌ | ❌ |
| 🎯 Adaptive Retrieval | ✅ **Built-in** | ❌ | ❌ | ❌ | ❌ |
| 📊 Pipeline Auto-Tuning | ✅ **Built-in** | ❌ | ❌ | ❌ | Prompts only |
| ✅ Answer Verification | ✅ **Built-in** | ❌ | ❌ | ❌ | ❌ |
| 🔗 Composable SmartPipeline | ✅ **Built-in** | ❌ | ❌ | ❌ | ❌ |
| 🛡️ Zero-Dep Guardrails | ✅ **Built-in** | External | ❌ | ❌ | ❌ |
| 🏘️ Community Detection | ✅ **Built-in** | ❌ | ❌ | ❌ | ❌ |

</td></tr>
</table>

---

<div align="center">

## ✨ Feature Overview

</div>

<table>
<tr>
<td width="50%" valign="top">

### 🧩 Core Pipeline
```
Loader → Chunker → Embedder → Retriever → Reranker → Generator
```
- **6 chunking strategies** — token, recursive, semantic, contextual, parent-child
- **6 embedder providers** — Ollama, SentenceTransformers, OpenAI, Voyage, Jina
- **7 retriever backends** — FAISS, NumPy, BM25, Hybrid RRF, ChromaDB, Qdrant, Graph
- **4 generator providers** — Ollama, OpenAI, Anthropic, LiteLLM (100+ models)
- Cross-encoder reranking · Query expansion (HyDE, Multi-Query, Step-Back)
- 10+ document loaders (PDF, DOCX, CSV, HTML, YouTube)

</td>
<td width="50%" valign="top">

### 🧠 Intelligence Layer
```
Query → Route → Retrieve → Verify → Respond
```
- **Knowledge Graph RAG** — entity extraction, community detection, graph+vector hybrid
- **Self-Correcting CRAG** — document grading → refine → web fallback
- **SelfRAG** — reflection tokens (relevance, support, usefulness)
- **ReAct Agent** — Think → Act → Observe loop with pluggable tools
- **Adaptive Retrieval** — auto strategy + confidence + fallback chain
- **SmartPipeline** — composable orchestrator wiring all modules
- Pipeline Optimizer — DSPy-inspired auto-tuning

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🛡️ Safety & Evaluation
```
Input → Guardrails → Process → Verify → Output
```
- **Answer Verifier** — claim-level hallucination detection & grounding
- **PII Redactor** — email, phone, SSN, credit card, IP detection
- **Injection Detector** — prompt injection with risk scoring
- **Topic Filter** — allowlist/blocklist topic restriction
- **LLM-as-Judge** — faithfulness, relevance, completeness (0–5)
- **9 metrics** — Hit Rate, MRR, P@K, R@K, NDCG, MAP, ROUGE-L, Context Precision

</td>
<td width="50%" valign="top">

### ⚡ Production Infrastructure
```
pip install ragpipe && python -m ragpipe serve
```
- **Async-first** — `aingest`, `aquery`, `aretrieve`, `stream_query`
- **Streaming** — real-time token streaming on every provider
- **FastAPI REST API** — full CRUD + WebSocket streaming
- **YAML config** — declarative pipeline definition
- **Semantic cache** — cosine similarity query cache + LRU embedding cache
- **Conversation memory** — multi-turn RAG with auto-contextualization
- **Observability** — structured tracing with per-step timing + JSON export

</td>
</tr>
</table>

<div align="center">

### 📊 By The Numbers

| Metric | Value |
|:-------|:------|
| **Tests** | 314 passed ✅ |
| **Test Time** | < 1 second |
| **Source Lines** | ~10,000 |
| **Core Dependencies** | 3 (numpy, rs-bpe, httpx) |
| **Optional Integrations** | 15+ providers |
| **Python** | 3.10+ |
| **License** | MIT |
| **Cloud Lock-in** | **Zero** — runs fully local with Ollama |

</div>

---

<div align="center">

## 📦 Install

</div>

```bash
pip install ragpipe                  # core: chunkers + retrievers + BM25
```

<details>
<summary>📦 <b>Optional extras — pick what you need</b></summary>

```bash
# LLM Providers
pip install 'ragpipe[openai]'        # OpenAI GPT-5.4 embedder & generator
pip install 'ragpipe[anthropic]'     # Anthropic Claude 4.6 generator
pip install 'ragpipe[litellm]'       # LiteLLM (100+ models: Gemini, Mistral, etc.)

# Vector Stores
pip install 'ragpipe[faiss]'         # FAISS (production, persistence)
pip install 'ragpipe[chroma]'        # ChromaDB (persistent local, metadata filtering)
pip install 'ragpipe[qdrant]'        # Qdrant (scalable cloud + self-hosted)

# Document Loaders
pip install 'ragpipe[data]'          # CSV/Excel (pandas)
pip install 'ragpipe[web]'           # HTML/Web (BeautifulSoup)
pip install 'ragpipe[youtube]'       # YouTube transcripts

# Infrastructure
pip install 'ragpipe[server]'        # FastAPI REST API + WebSocket server
pip install 'ragpipe[config]'        # YAML pipeline config

# Everything
pip install 'ragpipe[all]'           # all of the above
```

</details>

---

<div align="center">

## 🎯 Quickstart

</div>

### ⚡ 10-Line Hybrid Search Pipeline

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

---

### 🕸️ Knowledge Graph RAG

> Build a knowledge graph from your documents and query it with graph+vector hybrid retrieval.

```python
from ragpipe.graph import EntityExtractor, GraphBuilder, CommunityDetector, GraphRetriever

# 1. Extract entities and relationships from documents
extractor = EntityExtractor(extract_fn=my_llm)
builder = GraphBuilder(extractor=extractor)
result = builder.build(documents)

graph = result.graph
print(f"📊 Entities: {graph.entity_count}, Relationships: {graph.relationship_count}")

# 2. Detect communities for global search
detector = CommunityDetector(summarize_fn=my_llm)
communities = detector.detect(graph)

# 3. Graph-enhanced retrieval (local + global + vector hybrid)
retriever = GraphRetriever(
    graph=graph,
    communities=communities,
    vector_retrieve_fn=pipe.retrieve,
    generate_fn=my_llm,
    strategy="hybrid",          # "local", "global", or "hybrid"
    max_hops=2,
)
result = retriever.retrieve("How does Company X relate to Project Y?")
print(result.answer)
print(result.entities_used)         # entities traversed
print(result.communities_used)      # communities consulted
```

---

### 🪞 SelfRAG — Self-Reflective Generation

> The agent decides *whether* to retrieve, scores *relevance* of each passage, checks *support* for claims, and rates *usefulness* — then iterates if quality is low.

```python
from ragpipe.agents import SelfRAGAgent

agent = SelfRAGAgent(
    retrieve_fn=my_retrieve,
    generate_fn=my_generate,
    reflect_fn=my_llm,              # generates reflection tokens
    max_iterations=3,
)

result = agent.query("What is quantum computing?")
print(result.answer)
print(result.reflection.retrieval_needed)   # RetrievalDecision.RETRIEVE
print(result.reflection.support_level)      # SupportLevel.FULLY_SUPPORTED
print(result.reflection.usefulness)         # 5 (1–5 scale)
print(result.confidence)                    # 0.92
```

---

### ⚡ ReAct Agent — Reasoning + Tool Use

> Think → Act → Observe → Repeat. The agent reasons about what to do, executes tools, observes results, and iterates until it has a final answer.

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
print(result.steps)                 # [ReActStep(thought=..., action=..., observation=...), ...]
print(result.tools_used)            # ["search", "calculate"]
```

---

### 🔗 SmartPipeline — Composable Intelligence

> One `.query()` call orchestrates the entire intelligence stack: guardrails → PII → cache → memory → route → retrieve → verify → respond.

```python
from ragpipe.agents import SmartPipeline
from ragpipe.guardrails import PIIRedactor, PromptInjectionDetector, TopicGuardrail
from ragpipe.cache import SemanticCache
from ragpipe.memory import ConversationMemory
from ragpipe.verification import AnswerVerifier

smart = SmartPipeline(
    pipeline=my_pipeline,
    guardrails=[PromptInjectionDetector(), TopicGuardrail(allowed_topics=["finance"])],
    pii_redactor=PIIRedactor(),
    cache=SemanticCache(embed_fn=embed, threshold=0.95),
    memory=ConversationMemory(contextualize_fn=llm),
    verifier=AnswerVerifier(verify_fn=llm),
    on_guardrail_fail="block",
)

result = smart.query("What are the tax implications of stock options?")
print(result.answer)
print(result.guardrail_checks)      # {"injection": False, "topic": True}
print(result.verification)          # {"confidence": 0.95, "hallucination_rate": 0.0}
print(result.cached)                # False
print(result.latency_ms)            # 245.3
```

---

<details>
<summary><b>🔍 More Examples — CRAG, Adaptive, Router, Memory, Cache, Optimizer, Verifier, Guardrails, Evaluation, Tracing</b></summary>

### 🧠 Self-Correcting CRAG Agent

```python
from ragpipe.agents import CRAGAgent

agent = CRAGAgent(
    grade_fn=my_llm,              # grades each doc: CORRECT / AMBIGUOUS / INCORRECT
    retrieve_fn=my_retrieve,
    generate_fn=my_generate,
    web_search_fn=my_search,       # optional web fallback
)

result = agent.query("What caused the 2024 market correction?")
print(result.answer)               # grounded answer
print(result.action_taken)         # direct_generate / refined_generate / web_search / no_answer
print(result.confidence)           # 0.0–1.0
```

### 🎯 Adaptive Retrieval

```python
from ragpipe.agents import AdaptiveRetriever

retriever = AdaptiveRetriever(
    strategies={"dense": dense_fn, "sparse": sparse_fn, "hybrid": hybrid_fn},
    confidence_threshold=0.3,
)

result = retriever.retrieve("Compare FAISS vs ChromaDB performance")
print(result.strategy_used)         # RetrievalStrategy.MULTI_PASS
print(result.query_complexity)      # QueryComplexity.COMPARATIVE
```

### 🔀 Agentic RAG Router

```python
from ragpipe.agents import QueryRouter

router = QueryRouter(classify_fn=my_llm, retrieval_fn=my_retrieve, generation_fn=my_generate)
result = router.query("Compare the Q1 and Q2 financial results")
# Routes: direct answer / single retrieval / multi-step / summarize
```

### 💬 Conversation Memory

```python
from ragpipe.memory import ConversationMemory

memory = ConversationMemory(contextualize_fn=my_llm, max_history=20)
result = memory.query("What were the key findings?", retrieval_fn=my_pipeline)
result = memory.query("How does that compare to last quarter?", retrieval_fn=my_pipeline)
```

### 💾 Semantic Caching

```python
from ragpipe.cache import SemanticCache

cache = SemanticCache(embed_fn=my_embed, threshold=0.95, ttl=3600)
cached = cache.lookup("What is FAISS?")
```

### 📊 Pipeline Optimizer

```python
from ragpipe.optimization import PipelineOptimizer, ParameterSpace

optimizer = PipelineOptimizer(pipeline_factory=build_pipeline, eval_fn=evaluate, eval_dataset=qa_pairs)
result = optimizer.optimize(ParameterSpace(chunk_size=[256, 512, 1024], top_k=[3, 5, 10]))
print(result.best_params)           # {"chunk_size": 512, "top_k": 5}
```

### ✅ Answer Verification

```python
from ragpipe.verification import AnswerVerifier

verifier = AnswerVerifier(verify_fn=my_llm)
result = verifier.verify(answer="Paris is the capital.", sources=["Paris is the capital of France."])
print(result.hallucination_rate)     # 0.0
```

### 🛡️ Guardrails

```python
from ragpipe.guardrails import PIIRedactor, PromptInjectionDetector, TopicGuardrail

PIIRedactor().redact("Email john@test.com")              # "Email [EMAIL_REDACTED]"
PromptInjectionDetector().check("Ignore instructions").is_injection   # True
TopicGuardrail(allowed_topics=["finance"]).is_allowed("Tax question?")   # True
```

### ⚖️ LLM-as-Judge

```python
from ragpipe.evaluation import LLMJudge

judge = LLMJudge(judge_fn=my_llm)
scores = judge.evaluate(question="What is RAG?", answer=answer, context=context)
# {"faithfulness": 4.5, "relevance": 5.0, "completeness": 4.0, "overall": 4.5}
```

### 📊 Pipeline Observability

```python
from ragpipe.observability import Tracer

tracer = Tracer()
with tracer.span("retrieval", metadata={"top_k": 5}):
    results = pipe.retrieve("What is X?")
print(tracer.summary())             # per-step timing
print(tracer.to_json())             # structured JSON
```

</details>

---

<div align="center">

## 🏗️ Architecture

</div>

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                     ragpipe v3.0 — The Intelligent RAG Framework                    ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   INTELLIGENCE LAYER                                                                 ║
║   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    ║
║   │ SmartPipeline  │  │  SelfRAG       │  │  ReAct Agent   │  │  CRAG Agent    │    ║
║   │ ─────────────  │  │  ─────────     │  │  ──────────    │  │  ──────────    │    ║
║   │ Guardrails  →  │  │ IsRelevant? →  │  │ Think     →   │  │ Grade docs  →  │    ║
║   │ Cache       →  │  │ IsSupported?→  │  │ Act       →   │  │ Refine      →  │    ║
║   │ Memory      →  │  │ IsUseful?   →  │  │ Observe   →   │  │ Web search  →  │    ║
║   │ Route       →  │  │ Iterate?       │  │ Repeat        │  │ Generate       │    ║
║   │ Verify         │  │                │  │               │  │                │    ║
║   └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘    ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   CORE PIPELINE                                                                      ║
║   ┌────────┐  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌─────────┐  ┌──────────┐  ║
║   │ Loader │→ │ Chunker │→ │ Embedder │→ │ Retriever │→ │Reranker │→ │Generator │  ║
║   └────────┘  └─────────┘  └──────────┘  └───────────┘  └─────────┘  └──────────┘  ║
║   10+ types   6 strategies  6 providers   7 backends     CrossEnc.    4 providers    ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   CROSS-CUTTING SERVICES                                                             ║
║   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    ║
║   │Knowledge │ │ Semantic │ │ LLM-as-  │ │ Pipeline │ │  Query   │ │ Answer   │    ║
║   │  Graph   │ │  Cache   │ │  Judge   │ │  Tracer  │ │Expansion │ │ Verifier │    ║
║   ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤    ║
║   │Entities  │ │Query LRU │ │Faith.    │ │Spans     │ │HyDE      │ │Claims    │    ║
║   │Relations │ │Embedding │ │Relevance │ │Timing    │ │Multi-Q   │ │Grounding │    ║
║   │Community │ │TTL       │ │Complete  │ │JSON      │ │Step-Back │ │Halluc.   │    ║
║   └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

<div align="center">

## 📖 Components Reference

</div>

### Chunkers

| Chunker | Strategy | Best For |
|:--------|:---------|:---------|
| `TokenChunker` | Fixed token window + overlap | General purpose, predictable sizes |
| `RecursiveChunker` | Hierarchical separators (¶ → . → word) | Preserving document structure |
| `SemanticChunker` | Split at embedding similarity breakpoints | Maximum coherence per chunk |
| `ContextualChunker` | Prepend LLM-generated document context | 49% fewer retrieval failures |
| `ParentChildChunker` | Index children, return parent context | **Best precision + context** |

### Embedders

| Embedder | Local | Cost | Dimensions |
|:---------|:-----:|:-----|:----------:|
| `OllamaEmbedder` | ✅ | Free | 384–1024 |
| `SentenceTransformerEmbedder` | ✅ | Free | 384–1024 |
| `OpenAIEmbedder` | ☁️ | Paid | 1536–3072 |
| `VoyageEmbedder` | ☁️ | Paid | 512–1536 |
| `JinaEmbedder` | ☁️ | Paid | 512–1024 |

### Retrievers

| Retriever | Type | Best For |
|:----------|:-----|:---------|
| `NumpyRetriever` | Dense | Zero-dep, testing |
| `FaissRetriever` | Dense | Production, persistence |
| `ChromaRetriever` | Dense | Metadata filtering, persistent |
| `QdrantRetriever` | Dense | Cloud-scale + self-hosted |
| `BM25Retriever` | Sparse | Exact keyword matching |
| `HybridRetriever` | Dense + Sparse | **Best overall recall** |
| `GraphRetriever` | Graph + Vector | **Multi-hop reasoning** |

### Generators

| Generator | Local | Top Models (April 2026) |
|:----------|:-----:|:------------------------|
| `OllamaGenerator` | ✅ | gemma4, qwen3.5, llama4:scout, deepseek-v3.2 |
| `OpenAIGenerator` | ☁️ | gpt-5.4, gpt-5.4-pro, gpt-5.3-codex, gpt-5-mini |
| `AnthropicGenerator` | ☁️ | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |
| `LiteLLMGenerator` | Mixed | All above + gemini-3.1-pro, mistral-large-3 |

### Agents & Intelligence

| Agent | What It Does | Key Innovation |
|:------|:-------------|:---------------|
| `SmartPipeline` | Composable orchestrator | Guardrails → cache → memory → route → verify |
| `SelfRAGAgent` | Self-reflective generation | Reflection tokens (relevance, support, usefulness) |
| `ReActAgent` | Reasoning + tool use | Think → Act → Observe loop |
| `CRAGAgent` | Self-correcting retrieval | Document grading + refinement + web fallback |
| `AdaptiveRetriever` | Dynamic strategy selection | Query complexity → strategy → confidence → retry |
| `QueryRouter` | Routes queries to optimal strategy | 4 route types with decomposition |
| `GraphRetriever` | Graph-enhanced retrieval | Entity traversal + community search |

### Safety & Evaluation

| Module | What It Does |
|:-------|:-------------|
| `AnswerVerifier` | Claim-level hallucination detection |
| `PIIRedactor` | Email, phone, SSN, credit card, IP |
| `PromptInjectionDetector` | Injection patterns with risk scoring |
| `TopicGuardrail` | Allowlist/blocklist topic filtering |
| `LLMJudge` | Faithfulness, relevance, completeness |
| `PipelineOptimizer` | DSPy-inspired parameter tuning |

---

<div align="center">

## ⚡ Async & Streaming

</div>

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

<div align="center">

## 🌐 REST API Server

</div>

```bash
python -m ragpipe serve --config pipeline.yml --port 8000 --api-key mysecret
```

| Method | Path | Description |
|:-------|:-----|:------------|
| `POST` | `/ingest` | Ingest documents |
| `POST` | `/query` | RAG query → JSON |
| `WS` | `/query/stream` | WebSocket streaming |
| `GET` | `/stats` | Document & chunk counts |
| `DELETE` | `/index` | Clear index |
| `POST` | `/evaluate` | Run eval metrics |
| `GET` | `/health` | Health check |

---

<div align="center">

## 📋 YAML Pipeline Config

</div>

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

<div align="center">

## 🔌 Extending ragpipe

</div>

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

<div align="center">

## 🧪 Testing

</div>

```bash
pip install -e ".[dev]"
python run_tests.py              # 314 tests with per-test timing
pytest tests/ -v                 # standard pytest
```

---

<div align="center">

## 🗺️ Roadmap

</div>

See [ROADMAP.md](ROADMAP.md) for the complete transformation plan.

| Phase | Version | Status | Highlights |
|:------|:--------|:-------|:-----------|
| **Phase 1** — Production Foundation | v2.0 | ✅ | Async, streaming, REST API, 6 vector stores, YAML config |
| **Phase 2** — Intelligent Retrieval | v2.1 | ✅ | Agentic router, parent-child chunking, caching, memory, observability |
| **Phase 3** — Intelligence & Safety | v2.2 | ✅ | CRAG, adaptive retrieval, optimizer, verifier, guardrails |
| **Phase 4** — Knowledge & Agents | v3.0 | ✅ | Knowledge Graph RAG, SelfRAG, ReAct, SmartPipeline |
| **Phase 5** — Visual Platform | v3.1 | 🔜 | Gradio playground, visual pipeline builder, RAG analytics dashboard |
| **Phase 6** — Enterprise | v4.0 | 📋 | Multi-modal RAG, multi-agent collaboration, RAG-as-a-Service |

---

<div align="center">

## 📄 License

MIT — Use ragpipe freely in commercial and open-source projects.

---

**Built with ❤️ by [Shivam](https://github.com/shivamongit)**

*ragpipe — The Intelligent RAG Framework*

</div>
