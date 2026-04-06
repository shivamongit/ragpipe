# Changelog

All notable changes to ragpipe are documented here.

---

## [3.0.0] — April 2026

### Context Engineering Platform

ragpipe evolves from a RAG framework into a **Context Engineering Platform** — programmable context composition, knowledge graph fusion, agentic retrieval, self-improving pipelines, and simulation testing.

#### New Core Modules

- **Context Engineering** (`ragpipe.context`) — `ContextWindow` with programmable operators: add, deduplicate (exact/jaccard/hash), prioritize (relevance/diversity/recency/density/position), compress, filter, budget (token limits), resolve conflicts, and render. Replaces naive top-K stuffing with structured context composition.
- **Knowledge Graph RAG** (`ragpipe.graph`) — `KnowledgeGraph` with entity/relation extraction (LLM or heuristic), in-memory graph storage, multi-hop BFS search, neighbor lookup, and graph+vector fusion via RRF scoring.
- **Pipeline DAG** (`ragpipe.pipeline`) — `PipelineDAG` replaces linear pipelines with a directed acyclic graph supporting fan-out (parallel branches), fan-in (merge), conditional routing, edge conditions, topological execution, cycle detection, and per-node error handling.
- **Agentic Retrieval** (`ragpipe.agents.planner`) — `AgenticPipeline` with `RetrievalPlanner` (query decomposition into multi-step plans: search/compare/aggregate/verify), `RetrievalEvaluator` (quality scoring + retry decisions), and optional `critique_fn` for answer review.
- **Self-Improving Pipelines** (`ragpipe.optimization.self_improving`) — `SelfImprovingLoop` records quality feedback (LLM-Judge, user, metrics), tracks best-performing parameter configs, and suggests optimized parameters via random search, epsilon-greedy bandit, or Bayesian optimization (Optuna).
- **Dataset Intelligence** (`ragpipe.intelligence`) — `DatasetAnalyzer` detects empty/short documents, exact + near-duplicate content (Jaccard), stale documents (timestamp-based), low-quality/repetitive text, computes vocabulary richness and health score.
- **Retrieval Simulation** (`ragpipe.simulation`) — `SimulationRunner` is the "pytest for RAG": tests pipelines against 9 built-in failure scenarios (adversarial, missing context, contradictions, stale data, embedding noise, low relevance, empty retrieval, long queries, ambiguous queries) with custom assertions and latency thresholds.
- **Plugin System** (`ragpipe.plugins`) — `PluginRegistry` with explicit registration, setuptools entry point discovery, module scanning, category inference from class hierarchy, and `create()` factory method. 9 categories: chunker, embedder, retriever, generator, reranker, loader, agent, guardrail, evaluator.

#### Infrastructure Improvements

- **Retry/Backoff** (`ragpipe.utils.retry`) — `@retry` and `@aretry` decorators with exponential backoff, jitter, configurable max attempts/delay, retryable exception filtering (429/5xx for HTTP). Plus `retry_call()` / `aretry_call()` for non-decorator usage.
- **Cost Tracking** (`ragpipe.utils.costs`) — `CostTracker` with per-model pricing (OpenAI, Anthropic, Google, local models), budget enforcement, cost-by-model/operation breakdowns, and JSON export.
- **OpenTelemetry Export** (`ragpipe.observability.otel`) — `OTelExporter` converts ragpipe traces to OTLP format. Backends: console, JSON, OTLP HTTP, OTLP gRPC. Full OTLP JSON export for Jaeger/Grafana Tempo.
- **Auto-Tracing** — `Pipeline` now accepts optional `tracer=Tracer()` parameter and auto-traces all operations (ingest, embed, retrieve, rerank, generate) with per-step timing and metadata.
- **Batch Embedding** — `BaseEmbedder.embed_batch()` (sync) and `aembed_batch()` (async) with configurable `batch_size`, `max_concurrency`, and `on_progress` callback.
- **HybridRetriever Fix** — Pipeline now passes `query_text` through to `HybridRetriever.search()` so BM25 sparse search actually works (was silently degrading to dense-only).

#### CLI Expansion

- `ragpipe init` — scaffold a new RAG project with config, directories, and sample data
- `ragpipe ingest` — ingest documents from directory or files with verbose logging
- `ragpipe query` — interactive query with formatted output and timing
- `ragpipe eval` — run evaluation dataset and display metrics
- `ragpipe version` — show version info

#### Cookbooks

- `01_basic_rag.py` — Complete local RAG pipeline with Ollama
- `02_context_engineering.py` — Programmable context composition with ContextWindow
- `03_knowledge_graph_rag.py` — Entity extraction + graph search + fusion
- `04_simulation_testing.py` — "pytest for RAG" with failure scenarios
- `05_agentic_retrieval.py` — Multi-step query decomposition + critique
- `06_pipeline_dag.py` — Branching, conditional, and parallel workflows

#### Stats

- Test suite expanded from 215 to **431 tests** (all passing in < 1 second)
- 12 new modules, 8 new test files, 6 cookbook examples
- Zero new required dependencies (all new features use stdlib only)

---

## [2.2.0] — April 2026

### Phase 3: Intelligence & Safety Layer

#### Unique Differentiators (not available in LangChain, LlamaIndex, Haystack, or DSPy)

- **Self-Correcting RAG (CRAG)** (`ragpipe.agents.crag`) — `CRAGAgent` grades each retrieved document for relevance (CORRECT / AMBIGUOUS / INCORRECT), then decides: generate directly, refine knowledge, fall back to web search, or honestly return "I don't know". Based on the CRAG paper (Yan et al., 2024).
- **Adaptive Retrieval** (`ragpipe.agents.adaptive`) — `AdaptiveRetriever` classifies query complexity (factual / analytical / comparative / exploratory / conversational), auto-selects retrieval strategy (dense / sparse / hybrid / multi-pass), adjusts top_k dynamically, and retries with fallback chain on low confidence.
- **Pipeline Optimizer** (`ragpipe.optimization`) — `PipelineOptimizer` auto-tunes RAG parameters (chunk_size, top_k, overlap, thresholds) via grid or random search against your evaluation dataset. DSPy-inspired but for infrastructure parameters, not prompts.
- **Answer Verifier** (`ragpipe.verification`) — `AnswerVerifier` decomposes answers into individual claims, verifies each against source documents, computes per-claim confidence and overall hallucination rate, and outputs a grounded answer with only supported claims.
- **PII Redactor** (`ragpipe.guardrails.pii`) — Zero-dependency regex-based detection and redaction of email, phone, SSN, credit card, IP address, and date-of-birth patterns. Configurable type filtering.
- **Prompt Injection Detector** (`ragpipe.guardrails.injection`) — Pattern-matched detection of instruction override, role manipulation, system prompt extraction, delimiter injection, DAN/jailbreak, and encoding attacks. Weighted risk scoring with configurable threshold.
- **Topic Guardrail** (`ragpipe.guardrails.topic`) — Allowlist/blocklist topic restriction with keyword matching. Restrict your RAG to answer only domain-specific questions.

#### Improvements

- Test suite expanded from 131 to **215 tests** (all passing in < 0.5 seconds)
- 5 new test files: `test_crag.py`, `test_adaptive.py`, `test_optimizer.py`, `test_verifier.py`, `test_guardrails.py`
- 4 new modules: `ragpipe.agents.crag`, `ragpipe.agents.adaptive`, `ragpipe.optimization`, `ragpipe.verification`, `ragpipe.guardrails`

---

## [2.1.0] — April 2026

### Phase 2: Intelligent Retrieval & Agentic RAG

#### New Modules

- **Agentic RAG Router** (`ragpipe.agents.router`) — `QueryRouter` classifies queries into direct / single-retrieval / multi-step / summarize routes. Supports sync and async with parallel multi-step retrieval.
- **Parent-Child Chunker** (`ragpipe.chunkers.parent_child`) — `ParentChildChunker` indexes small child chunks for precise embedding, returns larger parent chunks for richer generation context. Word-based splitting with configurable overlap.
- **Semantic Cache** (`ragpipe.cache.semantic`) — `SemanticCache` deduplicates queries by cosine similarity (configurable threshold). Supports TTL expiry and max size limits.
- **Embedding Cache** (`ragpipe.cache.embedding`) — `EmbeddingCache` provides LRU caching for `embed()` calls, keyed by text content hash.
- **LLM-as-Judge** (`ragpipe.evaluation.llm_judge`) — `LLMJudge` scores RAG output on faithfulness, relevance, and completeness (0–5 scale) with configurable dimension weights.
- **Conversation Memory** (`ragpipe.memory.conversation`) — `ConversationMemory` manages multi-turn RAG with automatic follow-up question contextualization. Sync and async support.
- **Observability & Tracing** (`ragpipe.observability.tracer`) — `Tracer` with structured `Span` objects for per-step timing, metadata, error tracking. JSON export and `TracerCallback` for pipeline integration.

#### New Loaders

- **CSVLoader** (`ragpipe.loaders.csv_loader`) — loads CSV and Excel files via pandas. Configurable column selection, row-per-document or full-file modes.
- **HTMLLoader** (`ragpipe.loaders.html_loader`) — loads local HTML files and web URLs via BeautifulSoup. Extracts clean text with metadata.
- **YouTubeLoader** (`ragpipe.loaders.youtube_loader`) — loads YouTube video transcripts via `youtube-transcript-api`. Supports language selection.

#### Breaking Changes

- **tiktoken replaced with rs-bpe** — All chunkers (`TokenChunker`, `RecursiveChunker`, `SemanticChunker`) now use `rs-bpe` (Rust-based BPE tokenizer) instead of `tiktoken`. Same `cl100k_base` / `o200k_base` encodings. The `encoding` parameter is unchanged. This fixes persistent process hangs on Python 3.13.

#### Improvements

- Test suite expanded from 70 to **131 tests** (all passing in < 1 second)
- Added `run_tests.py` — in-process pytest runner with per-test timing report and slow-test detection
- New optional dependency groups: `data` (pandas), `web` (beautifulsoup4), `youtube` (youtube-transcript-api)
- `ParentChildChunker` added to `ragpipe.chunkers` exports

---

## [2.0.0] — April 2026

### Phase 1: Production Foundation

- **Async-first architecture** — every base class provides `asyncio.to_thread` defaults. Native async in HTTP providers.
- **Streaming generation** — `stream()` / `astream()` on all generators. Pipeline `stream_query()` for token-by-token output.
- **FastAPI REST API server** — `python -m ragpipe serve` with ingest, query, streaming (WebSocket), stats, evaluate endpoints. API key auth.
- **New generators** — `AnthropicGenerator` (Claude 4.6), `LiteLLMGenerator` (100+ models)
- **New embedders** — `VoyageEmbedder`, `JinaEmbedder`
- **New retrievers** — `ChromaRetriever`, `QdrantRetriever`
- **YAML pipeline config** — `PipelineConfig.from_yaml()` / `from_dict()` with component registry
- **April 2026 model updates** — GPT-5.4, Claude Opus/Sonnet 4.6, Gemini 3.1 Pro, DeepSeek V3.2

---

## [1.0.0] — Initial Release

- Core pipeline: `Document`, `Chunk`, `Pipeline` orchestrator
- Chunkers: `TokenChunker`, `RecursiveChunker`, `SemanticChunker`, `ContextualChunker`
- Embedders: `OllamaEmbedder`, `SentenceTransformerEmbedder`, `OpenAIEmbedder`
- Retrievers: `NumpyRetriever`, `FaissRetriever`, `BM25Retriever`, `HybridRetriever`
- Generators: `OllamaGenerator`, `OpenAIGenerator`
- Rerankers: `CrossEncoderReranker`
- Query expansion: `HyDEExpander`, `MultiQueryExpander`, `StepBackExpander`
- Evaluation: 9 metrics (Hit Rate, MRR, P@K, R@K, NDCG, MAP, ROUGE-L, Context Precision, Faithfulness)
- Loaders: `TextLoader`, `PDFLoader`, `DocxLoader`, `DirectoryLoader`
