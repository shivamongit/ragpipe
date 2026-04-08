# ragpipe Architecture — How It Works

This document explains ragpipe's internals: what each module does, how they connect, and why the design choices were made.

---

## What Is ragpipe?

ragpipe is a modular RAG (Retrieval-Augmented Generation) framework. It takes your documents, splits them into searchable chunks, indexes them for retrieval, and then uses an LLM to answer questions grounded in your actual data.

The key difference from basic RAG tutorials: ragpipe implements the **advanced patterns** that production systems use in 2026 — hybrid search, contextual chunking, query expansion, and proper evaluation.

---

## The Pipeline

Every ragpipe workflow follows this flow:

```
Documents → Chunker → Embedder → Retriever → Reranker → Generator → Answer
```

Each step is a pluggable component. You can swap any piece without touching the rest.

### Step 1: Document Loading

**What it does:** Reads files (PDF, DOCX, TXT, Markdown) and converts them into `Document` objects with content and metadata.

```python
# ragpipe/loaders/
TextLoader    → .txt, .md files
PDFLoader     → .pdf files (via PyPDF2)
DocxLoader    → .docx files (via python-docx)
DirectoryLoader → recursively loads all supported files from a folder
```

**Why separate loaders?** Different file formats need different parsers. The loader abstracts this away — downstream components only see `Document(content=str, metadata=dict)`.

### Step 2: Chunking

**What it does:** Splits large documents into smaller pieces that fit within embedding model context windows and are semantically coherent.

```python
# ragpipe/chunkers/
TokenChunker      → fixed-size windows by token count (512 tokens, 64 overlap)
RecursiveChunker  → tries paragraph → sentence → word boundaries before falling back
SemanticChunker   → embeds each sentence, splits where similarity drops below threshold
ContextualChunker → wraps any chunker + adds LLM-generated document context to each chunk
```

**Why multiple strategies?**

- **TokenChunker** is predictable and fast. Use when chunk size consistency matters (e.g., cost estimation).
- **RecursiveChunker** preserves document structure. Paragraphs stay together. A sentence about "the above table" stays near the table.
- **SemanticChunker** groups related sentences even across paragraph boundaries. Best coherence, but requires an embedder at chunking time.
- **ContextualChunker** is the biggest retrieval improvement. It calls an LLM to generate a 2-3 sentence context prefix for each chunk explaining where it fits in the document. Anthropic reported **49% fewer retrieval failures** with this approach. The chunk "Revenue was $4.2B" becomes "This section discusses Q3 2025 financial results from the annual report. Revenue was $4.2B" — dramatically easier to match.

### Step 3: Embedding

**What it does:** Converts text chunks into dense vector representations (arrays of floats). Similar texts have similar vectors.

```python
# ragpipe/embedders/
OllamaEmbedder              → local, free (nomic-embed-text, mxbai-embed-large, bge-m3)
SentenceTransformerEmbedder → local, free (all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5)
OpenAIEmbedder              → cloud API (text-embedding-3-small, text-embedding-3-large)
VoyageEmbedder              → cloud API (voyage-3, voyage-code-3, voyage-3-lite)
JinaEmbedder                → cloud API (jina-embeddings-v3 — 8192 token context)
```

All embedders support both `embed()` (sync) and `aembed()` (native async via httpx/AsyncOpenAI).

**Why Ollama-first?** Because it's free, private, and fast enough for most use cases. `nomic-embed-text` (768 dimensions) produces embeddings competitive with OpenAI's `text-embedding-3-small` at zero cost.

### Step 4: Retrieval

**What it does:** Given a query, finds the most relevant chunks from the index.

```python
# ragpipe/retrievers/
NumpyRetriever   → pure NumPy dot-product search, zero dependencies
FaissRetriever   → FAISS IndexFlatIP with L2-normalized cosine similarity + disk persistence
ChromaRetriever  → ChromaDB: persistent local store, metadata filtering, zero-config
QdrantRetriever  → Qdrant: self-hosted or cloud, metadata filtering, scalable
BM25Retriever    → Okapi BM25 keyword-based ranking (sparse retrieval)
HybridRetriever  → fuses dense + sparse results with Reciprocal Rank Fusion (RRF)
```

**Why hybrid search matters:**

Dense (vector) search understands meaning: "automobile" matches "car". But it misses exact keywords: searching for "error code E4021" won't match if the embedding doesn't capture that specific token.

Sparse (BM25) search captures exact keywords perfectly. "E4021" matches "E4021". But it misses synonyms: "automobile" won't match "car".

**HybridRetriever combines both** using Reciprocal Rank Fusion:

```
RRF_score(doc) = Σ weight / (k + rank)
```

A document ranked #1 in dense and #3 in BM25 scores higher than one ranked #2 in both. No score normalization needed — RRF works on ranks, not raw scores.

### Step 5: Query Expansion

**What it does:** Transforms the raw user query into better search queries before retrieval.

```python
# ragpipe/query/
HyDEExpander       → generates a hypothetical answer, searches for docs similar to that answer
MultiQueryExpander → generates N diverse reformulations of the question
StepBackExpander   → generates a broader question for background context
```

**Why this matters:**

User queries are often poor search queries. "Why is our API slow?" doesn't match well against documentation about "connection pool exhaustion" or "N+1 query patterns." Query expansion bridges this gap:

- **HyDE** generates "API slowness is typically caused by connection pool exhaustion, N+1 queries, or missing indexes..." — this hypothetical answer matches document language much better than the question.
- **Multi-Query** generates "API performance debugging", "slow response time causes", "backend latency troubleshooting" — covering terminology the user didn't think of.
- **Step-Back** generates "What are common causes of web application performance issues?" — retrieving background context that helps answer the specific question.

### Step 6: Reranking

**What it does:** Takes the top-K retrieved chunks and re-scores them with a more powerful (but slower) model.

```python
# ragpipe/rerankers/
CrossEncoderReranker → processes (query, chunk) pairs jointly for precise relevance scores
```

**Why rerank?** Bi-encoder retrieval (embedding similarity) is fast but approximate. A cross-encoder processes the query and passage together in a single forward pass, capturing token-level interactions. It's too slow to run on all chunks, but perfect for re-scoring the top 10-20 candidates down to the top 3-5.

### Step 7: Generation

**What it does:** Sends the query + retrieved context to an LLM to generate a cited answer.

```python
# ragpipe/generators/
OllamaGenerator    → local, free (gemma4, qwen3.5, llama4:scout, deepseek-v3.2, nemotron3)
OpenAIGenerator    → OpenAI (gpt-5.4, gpt-5.4-pro, gpt-5.3-codex, gpt-5-mini, gpt-5-nano)
AnthropicGenerator → Anthropic (claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5)
LiteLLMGenerator   → 100+ models (Gemini 3.1 Pro, Gemini 3 Flash, Mistral Large 3, DeepSeek V3.2, ...)
```

All generators support `generate()` / `agenerate()` (sync/async) and `stream()` / `astream()` (token-by-token streaming).

All generators use a structured system prompt that instructs the LLM to:
1. Answer only from the provided context
2. Cite sources using [Source N] notation
3. Admit when context is insufficient
4. Preserve technical terminology

### Step 8: Evaluation

**What it does:** Measures how good your retrieval and generation are.

```python
# ragpipe/evaluation/
# Retrieval metrics (do the right chunks come back?)
hit_rate()          → did we find at least one relevant chunk?
mrr()               → how early is the first relevant chunk?
precision_at_k()    → what fraction of top-K are relevant?
recall_at_k()       → what fraction of relevant docs are in top-K?
ndcg_at_k()         → rank-weighted quality score
map_at_k()          → average precision across all relevant positions
context_precision() → RAGAS-style weighted precision

# Generation metrics (is the answer good?)
rouge_l()           → longest common subsequence overlap with reference answer
faithfulness_score()→ n-gram overlap between answer and source chunks (grounding check)
```

---

## Data Flow Example

Here's what happens when you call `pipe.query("What is FAISS?")`:

```
1. embedder.embed(["What is FAISS?"])
   → [0.12, -0.34, 0.56, ...]  (768-dim vector)

2. dense_retriever.search(vector, top_k=15)
   → 15 chunks ranked by cosine similarity

3. bm25_retriever.search_text("What is FAISS?", top_k=15)
   → 15 chunks ranked by BM25 keyword score

4. hybrid_retriever.rrf_fuse(dense_results, sparse_results)
   → 5 chunks with combined RRF scores

5. reranker.rerank("What is FAISS?", 5_chunks, top_k=3)
   → 3 chunks re-scored by cross-encoder

6. generator.generate("What is FAISS?", 3_chunks)
   → "FAISS is a library developed by Meta for efficient similarity search
      of dense vectors [Source 1]. It supports both exact and approximate
      nearest neighbor search [Source 2]..."

7. Return GenerationResult(answer=..., sources=..., latency_ms=..., tokens_used=...)
```

---

## Project Structure

```
ragpipe/
├── ragpipe/
│   ├── __init__.py           # Package root, version, public API
│   ├── __main__.py           # CLI entry point (python -m ragpipe serve)
│   ├── core.py               # Document, Chunk, RetrievalResult, GenerationResult, Pipeline
│   │                           # (sync + async: ingest/aingest, query/aquery, stream_query)
│   ├── config.py             # YAML/dict pipeline configuration (PipelineConfig)
│   ├── server/
│   │   ├── __init__.py
│   │   └── app.py            # FastAPI REST API + WebSocket streaming
│   ├── chunkers/
│   │   ├── base.py           # Abstract BaseChunker
│   │   ├── token.py          # TokenChunker (rs-bpe)
│   │   ├── recursive.py      # RecursiveChunker (hierarchical separators)
│   │   ├── semantic.py       # SemanticChunker (embedding breakpoints)
│   │   └── contextual.py     # ContextualChunker (LLM context prefix)
│   ├── embedders/
│   │   ├── base.py           # Abstract BaseEmbedder (embed + aembed)
│   │   ├── ollama.py         # OllamaEmbedder (local, free)
│   │   ├── sentence_transformer.py  # SentenceTransformerEmbedder (local)
│   │   ├── openai.py         # OpenAIEmbedder (cloud API)
│   │   ├── voyage.py         # VoyageEmbedder (Voyage AI)
│   │   └── jina.py           # JinaEmbedder (Jina AI, 8192 ctx)
│   ├── retrievers/
│   │   ├── base.py           # Abstract BaseRetriever
│   │   ├── numpy_retriever.py    # NumpyRetriever (zero-dep)
│   │   ├── faiss_retriever.py    # FaissRetriever (IndexFlatIP + persistence)
│   │   ├── chroma_retriever.py   # ChromaRetriever (persistent local)
│   │   ├── qdrant_retriever.py   # QdrantRetriever (scalable cloud/self-hosted)
│   │   ├── bm25_retriever.py     # BM25Retriever (sparse keyword search)
│   │   └── hybrid_retriever.py   # HybridRetriever (RRF fusion)
│   ├── rerankers/
│   │   ├── base.py           # Abstract BaseReranker (rerank + arerank)
│   │   └── cross_encoder.py  # CrossEncoderReranker
│   ├── generators/
│   │   ├── base.py           # Abstract BaseGenerator (generate/agenerate/stream/astream)
│   │   ├── ollama_gen.py     # OllamaGenerator (local, free)
│   │   ├── openai_gen.py     # OpenAIGenerator (GPT-5.4, 5.3-Codex, etc.)
│   │   ├── anthropic_gen.py  # AnthropicGenerator (Claude Opus/Sonnet 4.6, Haiku 4.5)
│   │   └── litellm_gen.py    # LiteLLMGenerator (100+ models via single interface)
│   ├── query/
│   │   ├── __init__.py
│   │   └── expansion.py      # HyDEExpander, MultiQueryExpander, StepBackExpander
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py        # 9 retrieval & generation metrics
│   │   └── llm_judge.py      # LLM-as-Judge (faithfulness, relevance, completeness)
│   ├── loaders/
│   │   ├── text.py           # TextLoader (.txt, .md)
│   │   ├── pdf.py            # PDFLoader (.pdf)
│   │   ├── docx.py           # DocxLoader (.docx)
│   │   ├── csv_loader.py     # CSVLoader (.csv, .xlsx — pandas)
│   │   ├── html_loader.py    # HTMLLoader (.html, URLs — BeautifulSoup)
│   │   ├── youtube_loader.py # YouTubeLoader (transcript API)
│   │   └── directory.py      # DirectoryLoader (recursive)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router.py         # QueryRouter (direct/single/multi-step/summarize)
│   │   ├── crag.py           # CRAGAgent (self-correcting RAG with relevance grading)
│   │   ├── adaptive.py       # AdaptiveRetriever (auto strategy + confidence scoring)
│   │   ├── selfrag.py        # SelfRAGAgent (self-reflective retrieval tokens) ★ v3.0
│   │   ├── react.py          # ReActAgent (reasoning + acting with tool use) ★ v3.0
│   │   └── smart_pipeline.py # SmartPipeline (composable intelligence orchestrator) ★ v3.0
│   ├── graph/                # Knowledge Graph RAG ★ v3.0
│   │   ├── __init__.py
│   │   ├── entities.py       # Entity, Relationship, KnowledgeGraph, EntityExtractor
│   │   ├── builder.py        # GraphBuilder (build graph from documents)
│   │   ├── community.py      # CommunityDetector (label propagation)
│   │   └── retriever.py      # GraphRetriever (local/global/hybrid graph search)
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── semantic.py       # SemanticCache (cosine similarity threshold)
│   │   └── embedding.py      # EmbeddingCache (LRU, keyed by text hash)
│   ├── context/              #                                                           [v3.0]
│   │   ├── __init__.py
│   │   └── window.py         # ContextWindow (programmable context composition)
│   ├── graph/                #                                                           [v3.0]
│   │   ├── __init__.py
│   │   └── knowledge_graph.py # KnowledgeGraph (entity extraction, BFS, graph+vector fusion)
│   ├── pipeline/             #                                                           [v3.0]
│   │   ├── __init__.py
│   │   └── dag.py            # PipelineDAG (branching, conditional, parallel execution)
│   ├── plugins/              #                                                           [v3.0]
│   │   ├── __init__.py
│   │   └── registry.py       # PluginRegistry (entry points, discovery, factory)
│   ├── simulation/           #                                                           [v3.0]
│   │   ├── __init__.py
│   │   └── runner.py         # SimulationRunner ("pytest for RAG", 9 failure scenarios)
│   ├── intelligence/         #                                                           [v3.0]
│   │   ├── __init__.py
│   │   └── analyzer.py       # DatasetAnalyzer (staleness, duplicates, health scoring)
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── optimizer.py      # PipelineOptimizer (grid/random search, DSPy-inspired)
│   │   └── self_improving.py # SelfImprovingLoop (Bayesian/bandit optimization)          [v3.0]
│   ├── verification/
│   │   ├── __init__.py
│   │   └── verifier.py       # AnswerVerifier (claim decomposition, hallucination detection)
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── pii.py            # PIIRedactor (email, phone, SSN, credit card, IP)
│   │   ├── injection.py      # PromptInjectionDetector (pattern + risk scoring)
│   │   └── topic.py          # TopicGuardrail (allowlist/blocklist filtering)
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation.py   # ConversationMemory (multi-turn + contextualization)
│   ├── utils/                #                                                           [v3.0]
│   │   ├── __init__.py
│   │   ├── retry.py          # @retry / @aretry (exponential backoff, jitter)
│   │   └── costs.py          # CostTracker (per-model pricing, budget enforcement)
│   └── observability/
│       ├── __init__.py
│       └── tracer.py         # Tracer, Span, TracerCallback (structured tracing)
├── tests/                    # 314 tests covering all components
├── run_tests.py              # Test runner with per-test timing report
├── examples/                 # Quickstart and OpenAI pipeline examples
├── pyproject.toml            # Package config, dependencies, extras
├── CHANGELOG.md              # Release notes
├── ROADMAP.md                # Transformation roadmap
└── README.md                 # Documentation
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hybrid search by default** | Dense misses exact keywords, BM25 misses semantics. Combining both with RRF gives 10-30% better recall in benchmarks. |
| **Contextual chunking** | Anthropic's research shows 49% fewer retrieval failures. The upfront LLM cost pays for itself in accuracy. |
| **Ollama-first providers** | Local inference is free, private, and increasingly competitive with cloud APIs. |
| **FAISS IndexFlatIP** | Exact cosine similarity. No approximate index tuning needed under 100K vectors. |
| **rs-bpe cl100k_base** | Rust-based BPE tokenizer, faster than tiktoken with no network dependencies. Same cl100k_base encoding as GPT-4 and text-embedding-3. |
| **Async-first with sync compat** | All base classes default to `asyncio.to_thread` for free async. HTTP providers (Ollama, OpenAI, Anthropic) override with native async. Sync methods remain for simple scripts. |
| **Streaming generation** | Every generator supports `stream()` / `astream()`. Pipeline exposes `stream_query()` for token-by-token output. Server uses WebSocket streaming. |
| **REST API server** | `python -m ragpipe serve` spins up a FastAPI server with ingest, query, streaming, stats, and evaluation endpoints. API key auth included. |
| **YAML config** | `PipelineConfig.from_yaml()` enables declarative pipeline definition. Component registry maps type strings to classes. |
| **Optional dependencies** | Core needs only numpy + rs-bpe + httpx (~10 MB). Everything else (OpenAI, Anthropic, FAISS, ChromaDB, Qdrant) is opt-in. |
| **Abstract base classes** | Every component is a base class. Extend `BaseEmbedder` to add Cohere, Voyage AI, or any provider. |
| **Separate retrieve() and query()** | `retrieve()` returns chunks without generation — essential for evaluation, debugging, and hybrid workflows. |

---

## How to Extend

### Custom Embedder

```python
from ragpipe.embedders.base import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Your embedding logic
        return [[0.1, 0.2, ...] for _ in texts]

    @property
    def dim(self) -> int:
        return 768
```

### Custom Retriever

```python
from ragpipe.retrievers.base import BaseRetriever

class PineconeRetriever(BaseRetriever):
    def add(self, chunks, embeddings): ...
    def search(self, query_embedding, top_k=5): ...
    @property
    def count(self) -> int: ...
```

### Custom Generator

```python
from ragpipe.generators.base import BaseGenerator, GenerationOutput

class MyGenerator(BaseGenerator):
    def generate(self, question, context) -> GenerationOutput: ...
    async def agenerate(self, question, context) -> GenerationOutput: ...  # optional native async
    async def astream(self, question, context):  # optional streaming
        yield "token"
```

Every custom component plugs directly into `Pipeline`, the REST API server, and YAML config (via the component registry) without any other changes.
