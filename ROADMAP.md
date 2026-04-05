# ragpipe — Transformation Roadmap

> Last updated: April 2026 &middot; Current version: **v2.1.0**

---

## Current State

ragpipe is a production-grade, modular RAG framework with **131 tests** covering:

- **6 chunkers** — token, recursive, semantic, contextual, parent-child, custom
- **6 embedders** — Ollama, SentenceTransformers, OpenAI, Voyage, Jina, custom
- **6 retrievers** — FAISS, NumPy, BM25, Hybrid RRF, ChromaDB, Qdrant
- **4 generators** — Ollama, OpenAI GPT-5.4, Anthropic Claude 4.6, LiteLLM (100+)
- **10+ loaders** — PDF, DOCX, TXT, CSV/Excel, HTML/Web, YouTube, directory
- **Agentic router** — query classification + multi-step retrieval orchestration
- **Semantic caching** — query cache (cosine similarity) + LRU embedding cache
- **Conversation memory** — multi-turn RAG with automatic query contextualization
- **LLM-as-Judge** — faithfulness, relevance, completeness scoring (0–5 scale)
- **Observability** — structured tracing with per-step timing and JSON export
- **9 eval metrics** — Hit Rate, MRR, P@K, R@K, NDCG, MAP, ROUGE-L, Faithfulness
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

### Phase 3 — Intelligence Layer (Planned)

| # | Pillar | Notes |
|---|--------|-------|
| 13 | **Graph RAG** | Entity extraction, graph builder, community detection, graph+vector retrieval |
| 14 | **Guardrails & PII** | PII redaction, prompt injection detection, toxicity filter, output validation |
| — | **Full Agentic RAG** | CRAG (corrective), SelfRAG (self-reflective), ReAct agent loop |
| — | **UI & Deployment** | Gradio playground, Docker Compose, Helm chart |

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
| Graph RAG | ★★★☆☆ | High | P3 | Phase 3 |
| Guardrails & PII | ★★★☆☆ | Medium | P3 | Phase 3 |

---

## Release History

| Version | Date | Highlights |
|---------|------|------------|
| **v2.1.0** | April 2026 | Agentic router, parent-child chunking, semantic caching, conversation memory, LLM-as-Judge, observability, 10+ loaders, rs-bpe tokenizer, 131 tests |
| **v2.0.0** | April 2026 | Async-first, streaming, FastAPI server, Anthropic/LiteLLM/Voyage/Jina, ChromaDB/Qdrant, YAML config, April 2026 models |
| **v1.0.0** | — | Core pipeline: chunkers, embedders, retrievers, generators, evaluation |
