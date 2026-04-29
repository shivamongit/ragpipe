# Changelog

All notable changes to ragpipe are documented here.

---

## [3.1.0] — April 2026

### Production RAG Studio

ragpipe v3.1 turns the framework into a complete production-ready studio: multi-LLM provider support, polished Next.js UI, conversation persistence, and a one-command launcher.

#### New LLM Providers

- **GeminiGenerator** (`ragpipe.generators.gemini_gen`) — Google Gemini 2.5 Pro/Flash/Flash-Lite + 2.0 Flash via `google-genai` SDK. Native sync, async, and streaming.
- **GroqGenerator** (`ragpipe.generators.groq_gen`) — Groq's ultra-fast LPU inference for Llama 3.3 70B, Llama 3.1 8B Instant, Mixtral 8x7B, Qwen 3 32B, DeepSeek R1 Distill, Kimi K2.
- **CohereGenerator** (`ragpipe.generators.cohere_gen`) — Cohere Command A, Command R+/R/R7B with native RAG citation support via `cohere>=5.13`.
- **MistralGenerator** (`ragpipe.generators.mistral_gen`) — Mistral Large/Medium/Small + Codestral via `mistralai>=1.5`.

Total LLM providers: **8** (OpenAI, Anthropic, Gemini, Groq, Cohere, Mistral, Ollama, LiteLLM 100+).

#### Provider Registry & Runtime Switching

- **`ragpipe.generators.registry`** — Single source of truth for providers and models. `list_providers()`, `find_model()`, `build_generator()` for runtime model switching.
- **30+ curated models** with metadata: context window, pricing, tags (fast/cheap/reasoning), description.
- **Per-query overrides** — `/query` and `/query/stream` accept `provider`, `model`, `api_key_override` so a single backend can talk to any LLM.

#### Server Upgrades

- **`GET /providers`** — Lists all providers with installed/key-set availability.
- **`GET /models`** — Lists all models with `?provider=` and `?available_only=` filters.
- **`POST /upload`** — Multipart file uploads with auto-extraction (PDF, DOCX, TXT, MD, HTML, CSV).
- **Conversation persistence** — SQLite-backed `/conversations` CRUD endpoints. Messages, sources, tokens, latency stored per conversation.
- **Streaming sources** — WebSocket streams now emit a `sources` frame before tokens.

#### Production Polish

- **`launch.py`** — Auto-detect best available provider, build pipeline, start uvicorn. Loads `.env` automatically.
- **`start.sh`** — One-command launcher. Installs deps, starts backend (8000) + Next.js UI (3000) together.
- **`.env.example`** — Documented environment variables for all providers.

#### UI Studio (Next.js 16 + Tailwind v4)

- **Sidebar** with grouped conversation history (Today / Yesterday / Previous 7 days / Previous 30 days / Older), inline rename, and delete.
- **Model picker** — Searchable dropdown of all 30+ models, grouped by provider, with pricing/context/tag badges and inline availability indicator.
- **Settings modal** — Tabbed UI for per-provider API keys, server URL, and retrieval params. Keys stored in `localStorage`.
- **Ingest panel** — Drag-and-drop or paste, with current document/chunk count and clear-all action.
- **Streaming responses** — Token-by-token assistant responses with optimistic UI and stop button.
- **Source citations** — Expandable cards with relevance scores and full text.
- **Markdown rendering** — Lightweight in-browser renderer (headings, lists, code, blockquotes, tables, source pills).

#### Tests

- **+23 new tests** for the registry, conversation storage, and server upgrades.
- Total: **454 tests passing** in < 1s.

---

## [3.0.0] — April 2026

### Phase 4: Knowledge Graph & Advanced Agents

#### Unique Differentiators (not available in LangChain, LlamaIndex, Haystack, or DSPy)

- **Knowledge Graph RAG** (`ragpipe.graph`) — `EntityExtractor` extracts entities and relationships from text (LLM or regex heuristics). `GraphBuilder` builds a `KnowledgeGraph` from documents with entity deduplication. `CommunityDetector` runs label propagation to find graph communities. `GraphRetriever` combines local graph traversal, global community search, and vector retrieval into a hybrid strategy.
- **SelfRAG Agent** (`ragpipe.agents.selfrag`) — `SelfRAGAgent` implements self-reflective retrieval with four reflection tokens: `IsRetrievalNeeded`, `IsRelevant`, `IsSupported`, `IsUseful`. Iterates on low-quality answers. Based on the SelfRAG paper (Asai et al., 2023).
- **ReAct Agent** (`ragpipe.agents.react`) — `ReActAgent` implements the Reasoning + Acting paradigm with pluggable `Tool` objects. Think → Act → Observe → Repeat loop with configurable max steps.
- **SmartPipeline** (`ragpipe.agents.smart_pipeline`) — `SmartPipeline` composes all ragpipe intelligence modules into a single `.query()` call: guardrails → cache → memory → route → retrieve → verify → respond.

#### Improvements

- Test suite expanded from 215 to **314 tests** (all passing in < 1 second)
- 7 new test files: `test_graph_entities.py`, `test_graph_builder.py`, `test_graph_community.py`, `test_graph_retriever.py`, `test_selfrag.py`, `test_react.py`, `test_smart_pipeline.py`
- 5 new modules: `ragpipe.graph` (4 files), `ragpipe.agents.selfrag`, `ragpipe.agents.react`, `ragpipe.agents.smart_pipeline`
- Premium README with competitive comparison, architecture diagrams, comprehensive code examples

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
