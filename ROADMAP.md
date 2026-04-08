# ragpipe — Transformation Roadmap

> Last updated: April 2026 &middot; Current version: **v3.0.0**

---

## Current State

ragpipe is a production-grade, modular RAG framework with **314 tests** covering:

- **6 chunkers** — token, recursive, semantic, contextual, parent-child, custom
- **6 embedders** — Ollama, SentenceTransformers, OpenAI, Voyage, Jina, custom
- **7 retrievers** — FAISS, NumPy, BM25, Hybrid RRF, ChromaDB, Qdrant, **GraphRetriever**
- **4 generators** — Ollama, OpenAI GPT-5.4, Anthropic Claude 4.6, LiteLLM (100+)
- **10+ loaders** — PDF, DOCX, TXT, CSV/Excel, HTML/Web, YouTube, directory
- **Knowledge Graph RAG** — entity extraction, graph builder, community detection, graph+vector hybrid retrieval
- **6 agents** — query router, CRAG (self-correcting), SelfRAG (self-reflective), ReAct (tool use), adaptive retrieval, SmartPipeline (composable)
- **Semantic caching** — query cache (cosine similarity) + LRU embedding cache
- **Conversation memory** — multi-turn RAG with automatic query contextualization
- **Answer verification** — hallucination detection, claim-level grounding, confidence scores
- **Pipeline optimizer** — DSPy-inspired auto-tuning of chunk_size, top_k, overlap
- **Guardrails** — PII redaction, prompt injection detection, topic filtering
- **LLM-as-Judge** — faithfulness, relevance, completeness scoring (0–5 scale)
- **Observability** — structured tracing with per-step timing and JSON export
- Async-first architecture, streaming generation, FastAPI server, YAML config
- Latest April 2026 models throughout (GPT-5.4, Claude 4.6, Gemini 3.1)

---

## The 14 Pillars

### Phase 1 — Production Foundation ✅ `v2.0.0`

| # | Pillar | Status |
|---|--------|--------|
| 1 | **Async-First Architecture** — `aingest()`, `aquery()`, `aretrieve()`, `stream_query()` | ✅ Complete |
| 2 | **Streaming Generation** — `stream()` / `astream()` on every generator, WebSocket support | ✅ Complete |
| 3 | **REST API Server + CLI** — `python -m ragpipe serve` with FastAPI | ✅ Complete |
| 4 | **LLM Integrations** — Anthropic, LiteLLM (100+), Ollama, OpenAI | ✅ Complete |
| 5 | **Vector Stores** — FAISS, ChromaDB, Qdrant, NumPy, BM25, Hybrid | ✅ Complete |
| 6 | **YAML Pipeline Config** — declarative `PipelineConfig.from_yaml()` | ✅ Complete |

### Phase 2 — Intelligent Retrieval & Agentic RAG ✅ `v2.1.0`

| # | Pillar | Module | Status |
|---|--------|--------|--------|
| 7 | **Agentic RAG Router** — query classification → direct/single/multi-step/summarize | `ragpipe.agents.router` | ✅ Complete |
| 8 | **Parent-Child Chunking** — index small children, return large parent context | `ragpipe.chunkers.parent_child` | ✅ Complete |
| 9 | **Semantic Caching** — query cache (cosine > threshold) + LRU embedding cache | `ragpipe.cache` | ✅ Complete |
| 10 | **LLM-as-Judge Evaluation** — faithfulness, relevance, completeness (0–5 scale) | `ragpipe.evaluation.llm_judge` | ✅ Complete |
| 11 | **Document Loaders** — CSV/Excel, HTML/Web, YouTube transcripts | `ragpipe.loaders` | ✅ Complete |
| 12 | **Observability & Tracing** — structured spans, per-step timing, JSON export | `ragpipe.observability` | ✅ Complete |
| — | **Conversation Memory** — multi-turn RAG with auto-contextualization | `ragpipe.memory` | ✅ Complete |
| — | **rs-bpe Migration** — replaced tiktoken with Rust-based BPE tokenizer | all chunkers | ✅ Complete |

### Phase 3 — Intelligence & Safety Layer ✅ `v2.2.0`

| # | Pillar | Module | Status |
|---|--------|--------|--------|
| 13 | **Self-Correcting RAG (CRAG)** — relevance grading → refine / web-search / no-answer | `ragpipe.agents.crag` | ✅ Complete |
| 14 | **Adaptive Retrieval** — query classification → auto strategy + top_k + fallback chain | `ragpipe.agents.adaptive` | ✅ Complete |
| 15 | **Pipeline Optimizer** — DSPy-inspired grid/random search over chunk_size, top_k, overlap | `ragpipe.optimization` | ✅ Complete |
| 16 | **Answer Verifier** — claim decomposition, per-claim grounding, hallucination rate | `ragpipe.verification` | ✅ Complete |
| 17 | **Guardrails: PII Redactor** — regex-based detection of email, phone, SSN, credit card, IP | `ragpipe.guardrails.pii` | ✅ Complete |
| 18 | **Guardrails: Injection Detector** — pattern-matched prompt injection with risk scoring | `ragpipe.guardrails.injection` | ✅ Complete |
| 19 | **Guardrails: Topic Filter** — allowlist/blocklist topic restriction with keyword matching | `ragpipe.guardrails.topic` | ✅ Complete |

### Phase 4 — Knowledge Graph & Advanced Agents ✅ `v3.0.0`

| # | Pillar | Module | Status |
|---|--------|--------|--------|
| 20 | **Knowledge Graph RAG** — entity extraction, graph builder, community detection, graph+vector hybrid | `ragpipe.graph` | ✅ Complete |
| 21 | **SelfRAG Agent** — self-reflective retrieval with relevance/support/usefulness tokens | `ragpipe.agents.selfrag` | ✅ Complete |
| 22 | **ReAct Agent** — reasoning + acting loop with pluggable tool use | `ragpipe.agents.react` | ✅ Complete |
| 23 | **SmartPipeline** — composable intelligence orchestrator (guardrails → cache → memory → route → verify) | `ragpipe.agents.smart_pipeline` | ✅ Complete |

### Phase 5 — Visual Platform *(Planned, v3.1)*

| # | Pillar | Notes |
|---|--------|-------|
| 24 | **Gradio RAG Playground** | Interactive query testing with source highlighting, confidence scores, trace visualization |
| 25 | **Visual Pipeline Builder** | Drag-and-drop pipeline design: connect components visually → export YAML → run |
| 26 | **Knowledge Graph Visualizer** | Interactive graph exploration with entity clustering, relationship mapping, community views |
| 27 | **RAG Analytics Dashboard** | Real-time metrics: latency, cache hit rate, hallucination rate, cost tracking |
| 28 | **Docker & Helm** | Docker Compose for local, Helm chart for Kubernetes |
| 29 | **Benchmark Suite** | Automated benchmarks on HotpotQA, NaturalQuestions, MMLU |
| 30 | **Plugin System** | Setuptools entry points for community-contributed components |

### Phase 6 — Enterprise Intelligence *(Planned, v4.0)*

| # | Pillar | Notes |
|---|--------|-------|
| 31 | **Multi-Modal RAG** | Images, tables, audio, video in the retrieval pipeline |
| 32 | **Multi-Agent Collaboration** | Specialized agents (researcher, analyst, verifier) collaborating on queries |
| 33 | **Context Engineering** | Programmable context windows: compress, deduplicate, prioritize, budget |
| 34 | **Self-Improving Pipeline** | Online learning loop: query → feedback → auto-tune → improve |
| 35 | **Advanced Knowledge Graph** | Temporal graphs, hierarchical entities, cross-document resolution |
| 36 | **RAG Security** | RBAC, audit logging, encryption at rest, compliance (SOC2, HIPAA) |

### Phase 7 — Platform & Ecosystem *(Planned, v5.0)*

| # | Pillar | Notes |
|---|--------|-------|
| 37 | **RAG-as-a-Service** | Hosted ragpipe with API keys, billing, team management |
| 38 | **Component Marketplace** | Community-contributed components with ratings and versioning |
| 39 | **TypeScript SDK** | JS/TS client + React components for ragpipe API |
| 40 | **Migration Tooling** | `ragpipe migrate --from langchain` automatic translation |
| 41 | **Model Fine-Tuning** | Built-in fine-tuning on domain-specific data |

---

## Priority & Impact Matrix

| Pillar | Impact | Effort | Priority | Status |
|--------|--------|--------|----------|--------|
| Async-First Architecture | ★★★★★ | Medium | P0 | ✅ v2.0 |
| Streaming Generation | ★★★★★ | Low | P0 | ✅ v2.0 |
| REST API Server + CLI | ★★★★★ | Medium | P0 | ✅ v2.0 |
| LLM Integrations | ★★★★★ | Low | P1 | ✅ v2.0 |
| Vector Stores | ★★★★★ | Medium | P1 | ✅ v2.0 |
| YAML Pipeline Config | ★★★★☆ | Low | P1 | ✅ v2.0 |
| Agentic RAG Router | ★★★★★ | High | P1 | ✅ v2.1 |
| Parent-Child Chunking | ★★★★☆ | Medium | P1 | ✅ v2.1 |
| Semantic Caching | ★★★★☆ | Medium | P2 | ✅ v2.1 |
| LLM-as-Judge Eval | ★★★★☆ | Medium | P2 | ✅ v2.1 |
| Document Loaders | ★★★★☆ | Low | P2 | ✅ v2.1 |
| Observability & Tracing | ★★★★☆ | Medium | P2 | ✅ v2.1 |
| Conversation Memory | ★★★★☆ | Medium | P2 | ✅ v2.1 |
| CRAG (Self-Correcting) | ★★★★★ | High | P1 | ✅ v2.2 |
| Adaptive Retrieval | ★★★★★ | Medium | P1 | ✅ v2.2 |
| Pipeline Optimizer | ★★★★★ | Medium | P1 | ✅ v2.2 |
| Answer Verifier | ★★★★★ | Medium | P1 | ✅ v2.2 |
| Guardrails (PII/Injection/Topic) | ★★★★☆ | Medium | P2 | ✅ v2.2 |
| Knowledge Graph RAG | ★★★★★ | High | P1 | ✅ v3.0 |
| SelfRAG Agent | ★★★★★ | High | P1 | ✅ v3.0 |
| ReAct Agent | ★★★★☆ | Medium | P2 | ✅ v3.0 |
| SmartPipeline | ★★★★★ | Medium | P1 | ✅ v3.0 |
| Gradio Playground | ★★★★★ | Medium | P1 | Phase 5 |
| Visual Pipeline Builder | ★★★★★ | High | P1 | Phase 5 |
| Knowledge Graph Visualizer | ★★★★★ | Medium | P2 | Phase 5 |
| RAG Analytics Dashboard | ★★★★☆ | Medium | P2 | Phase 5 |
| Docker & Helm | ★★★★☆ | Low | P2 | Phase 5 |
| Benchmarks | ★★★★☆ | Medium | P2 | Phase 5 |
| Plugin System | ★★★☆☆ | Medium | P3 | Phase 5 |
| Multi-Modal RAG | ★★★★★ | High | P1 | Phase 6 |
| Multi-Agent Collaboration | ★★★★★ | High | P1 | Phase 6 |
| Context Engineering | ★★★★★ | High | P1 | Phase 6 |
| Self-Improving Pipeline | ★★★★☆ | High | P2 | Phase 6 |
| RAG-as-a-Service | ★★★★★ | Very High | P1 | Phase 7 |

---

## Release History

| Version | Date | Highlights |
|---------|------|------------|
| **v3.0.0** | April 2026 | Knowledge Graph RAG (entity extraction, graph builder, community detection, graph+vector retrieval), SelfRAG agent (self-reflective tokens), ReAct agent (tool use), SmartPipeline (composable orchestrator), 314 tests |
| **v2.2.0** | April 2026 | CRAG agent, adaptive retrieval, pipeline optimizer, answer verifier, guardrails (PII/injection/topic), 215 tests |
| **v2.1.0** | April 2026 | Agentic router, parent-child chunking, semantic caching, conversation memory, LLM-as-Judge, observability, 10+ loaders, rs-bpe tokenizer, 131 tests |
| **v2.0.0** | April 2026 | Async-first, streaming, FastAPI server, Anthropic/LiteLLM/Voyage/Jina, ChromaDB/Qdrant, YAML config, April 2026 models |
| **v1.0.0** | — | Core pipeline: chunkers, embedders, retrievers, generators, evaluation |
