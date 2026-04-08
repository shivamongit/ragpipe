"""CLI entry point for ragpipe.

Commands:
    python -m ragpipe init [dir]               — scaffold a new ragpipe project
    python -m ragpipe ingest --config p.yml --dir ./docs
    python -m ragpipe query --config p.yml "What is RAG?"
    python -m ragpipe eval --config p.yml --dataset qa.json
    python -m ragpipe serve --port 8000 --config p.yml
    python -m ragpipe version
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="ragpipe",
        description="ragpipe — Production-grade RAG framework CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # --- init ---
    init_p = sub.add_parser("init", help="Scaffold a new ragpipe project")
    init_p.add_argument("directory", nargs="?", default=".", help="Project directory (default: .)")
    init_p.add_argument("--template", default="default", help="Project template (default/minimal)")

    # --- ingest ---
    ingest_p = sub.add_parser("ingest", help="Ingest documents into a pipeline")
    ingest_p.add_argument("--config", required=True, help="Path to pipeline YAML config")
    ingest_p.add_argument("--dir", default=None, help="Directory of documents to ingest")
    ingest_p.add_argument("--file", default=None, action="append", help="Individual file(s) to ingest")
    ingest_p.add_argument("--glob", default="**/*.txt", help="Glob pattern for directory ingestion")
    ingest_p.add_argument("--verbose", action="store_true", help="Show detailed progress")

    # --- query ---
    query_p = sub.add_parser("query", help="Query a pipeline")
    query_p.add_argument("question", help="The question to ask")
    query_p.add_argument("--config", required=True, help="Path to pipeline YAML config")
    query_p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    query_p.add_argument("--retrieve-only", action="store_true", help="Retrieve without generation")
    query_p.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    query_p.add_argument("--verbose", action="store_true", help="Show sources and metadata")

    # --- eval ---
    eval_p = sub.add_parser("eval", help="Evaluate a pipeline against a QA dataset")
    eval_p.add_argument("--config", required=True, help="Path to pipeline YAML config")
    eval_p.add_argument("--dataset", required=True, help="Path to QA dataset (JSON/JSONL)")
    eval_p.add_argument("--metrics", default="hit_rate,mrr,ndcg", help="Comma-separated metrics")
    eval_p.add_argument("--top-k", type=int, default=5, help="Retrieval top_k")
    eval_p.add_argument("--output", default=None, help="Save results to JSON file")
    eval_p.add_argument("--verbose", action="store_true", help="Show per-query details")

    # --- serve ---
    serve_p = sub.add_parser("serve", help="Start the ragpipe REST API server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_p.add_argument("--config", default=None, help="Path to pipeline YAML config file")
    serve_p.add_argument("--api-key", default=None, help="API key for authentication")
    serve_p.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # --- version ---
    sub.add_parser("version", help="Show ragpipe version")

    args = parser.parse_args()

    commands = {
        "init": _init,
        "ingest": _ingest,
        "query": _query,
        "eval": _eval,
        "serve": _serve,
        "version": _version,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


# ── init ──────────────────────────────────────────────────────────────────────

_PIPELINE_YAML_TEMPLATE = """\
# ragpipe pipeline configuration
# Docs: https://github.com/shivamongit/ragpipe

chunker:
  type: recursive
  chunk_size: 512
  overlap: 64

embedder:
  type: ollama
  model: nomic-embed-text

retriever:
  type: hybrid
  dense: {{type: numpy}}
  sparse: {{type: bm25}}

generator:
  type: ollama
  model: gemma4

# Optional:
# reranker:
#   type: cross_encoder
#   model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

_README_TEMPLATE = """\
# My RAGpipe Project

Built with [ragpipe](https://github.com/shivamongit/ragpipe) — production-grade RAG framework.

## Quick Start

```bash
# Ingest documents
python -m ragpipe ingest --config pipeline.yml --dir ./docs

# Query
python -m ragpipe query --config pipeline.yml "What is the main finding?"

# Start API server
python -m ragpipe serve --config pipeline.yml
```
"""


def _init(args):
    directory = os.path.abspath(args.directory)
    os.makedirs(directory, exist_ok=True)

    docs_dir = os.path.join(directory, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    config_path = os.path.join(directory, "pipeline.yml")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(_PIPELINE_YAML_TEMPLATE)

    readme_path = os.path.join(directory, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(_README_TEMPLATE)

    sample_doc = os.path.join(docs_dir, "example.txt")
    if not os.path.exists(sample_doc):
        with open(sample_doc, "w") as f:
            f.write("This is a sample document. Replace it with your own data.\n")

    print(f"✅ Initialized ragpipe project in {directory}")
    print(f"   📄 pipeline.yml — pipeline configuration")
    print(f"   📁 docs/        — place your documents here")
    print(f"   📖 README.md    — getting started guide")
    print()
    print("Next steps:")
    print(f"  1. Add documents to {docs_dir}/")
    print(f"  2. python -m ragpipe ingest --config {config_path} --dir {docs_dir}")
    print(f"  3. python -m ragpipe query --config {config_path} \"Your question\"")


# ── ingest ────────────────────────────────────────────────────────────────────

def _ingest(args):
    from ragpipe.config import PipelineConfig
    from ragpipe.core import Document

    pipeline = PipelineConfig.from_yaml(args.config).build()

    documents = []

    if args.dir:
        import glob
        pattern = os.path.join(args.dir, args.glob)
        paths = sorted(glob.glob(pattern, recursive=True))
        for path in paths:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        documents.append(Document(
                            content=content,
                            metadata={"source": path, "filename": os.path.basename(path)},
                        ))
                        if args.verbose:
                            print(f"  📄 {path} ({len(content):,} chars)")
                except Exception as e:
                    print(f"  ⚠️  Skipped {path}: {e}", file=sys.stderr)

    if args.file:
        for path in args.file:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    documents.append(Document(
                        content=content,
                        metadata={"source": path, "filename": os.path.basename(path)},
                    ))
            except Exception as e:
                print(f"  ⚠️  Skipped {path}: {e}", file=sys.stderr)

    if not documents:
        print("❌ No documents found to ingest.")
        sys.exit(1)

    print(f"Ingesting {len(documents)} document(s)...")
    t0 = time.perf_counter()
    stats = pipeline.ingest(documents)
    elapsed = time.perf_counter() - t0

    print(f"✅ Ingested {stats['documents']} documents → {stats['chunks']} chunks in {elapsed:.2f}s")


# ── query ─────────────────────────────────────────────────────────────────────

def _query(args):
    from ragpipe.config import PipelineConfig

    pipeline = PipelineConfig.from_yaml(args.config).build()

    if args.retrieve_only:
        results = pipeline.retrieve(args.question, top_k=args.top_k)
        if args.json_output:
            data = [{"text": r.chunk.text, "score": r.score, "rank": r.rank,
                      "metadata": r.chunk.metadata} for r in results]
            print(json.dumps(data, indent=2))
        else:
            print(f"Retrieved {len(results)} chunks:\n")
            for r in results:
                print(f"  [{r.rank}] (score={r.score:.4f}) {r.chunk.text[:120]}...")
                if args.verbose and r.chunk.metadata:
                    print(f"      metadata: {r.chunk.metadata}")
                print()
    else:
        result = pipeline.query(args.question, top_k=args.top_k)
        if args.json_output:
            data = {
                "answer": result.answer,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "tokens_used": result.tokens_used,
                "sources": [{"text": s.chunk.text[:200], "score": s.score}
                            for s in result.sources],
            }
            print(json.dumps(data, indent=2))
        else:
            print(f"\n{result.answer}\n")
            if args.verbose:
                print(f"--- Model: {result.model} | Latency: {result.latency_ms:.0f}ms | "
                      f"Tokens: {result.tokens_used} ---")
                print(f"Sources ({len(result.sources)}):")
                for s in result.sources:
                    print(f"  [{s.rank}] {s.chunk.text[:100]}...")


# ── eval ──────────────────────────────────────────────────────────────────────

def _eval(args):
    from ragpipe.config import PipelineConfig

    pipeline = PipelineConfig.from_yaml(args.config).build()

    # Load dataset (JSON array of {"question": ..., "answer": ..., "context": ...})
    with open(args.dataset, "r") as f:
        content = f.read().strip()
        if content.startswith("["):
            dataset = json.loads(content)
        else:
            # JSONL format
            dataset = [json.loads(line) for line in content.splitlines() if line.strip()]

    metric_names = [m.strip() for m in args.metrics.split(",")]
    results = {"metrics": {}, "per_query": [], "total_queries": len(dataset)}

    from ragpipe.evaluation.metrics import (
        hit_rate, mean_reciprocal_rank, precision_at_k, recall_at_k, ndcg_at_k,
    )

    metric_fns = {
        "hit_rate": hit_rate,
        "mrr": mean_reciprocal_rank,
        "precision": precision_at_k,
        "recall": recall_at_k,
        "ndcg": ndcg_at_k,
    }

    scores = {m: [] for m in metric_names}

    print(f"Evaluating {len(dataset)} queries with metrics: {', '.join(metric_names)}")
    print()

    for i, item in enumerate(dataset):
        question = item.get("question", item.get("query", ""))
        expected = item.get("answer", item.get("expected", ""))
        relevant_ids = item.get("relevant_ids", [])

        retrieved = pipeline.retrieve(question, top_k=args.top_k)
        retrieved_ids = [r.chunk.id for r in retrieved]
        retrieved_texts = [r.chunk.text for r in retrieved]

        query_scores = {}
        for m in metric_names:
            fn = metric_fns.get(m)
            if fn and relevant_ids:
                try:
                    s = fn(relevant_ids, retrieved_ids, k=args.top_k)
                    scores[m].append(s)
                    query_scores[m] = s
                except Exception:
                    pass

        if args.verbose:
            print(f"  [{i+1}/{len(dataset)}] {question[:60]}... → {query_scores}")

        results["per_query"].append({
            "question": question,
            "scores": query_scores,
            "retrieved_count": len(retrieved),
        })

    # Aggregate
    for m in metric_names:
        vals = scores.get(m, [])
        results["metrics"][m] = round(sum(vals) / len(vals), 4) if vals else 0.0

    print()
    print("=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for m, v in results["metrics"].items():
        print(f"  {m:>15s}: {v:.4f}")
    print("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


# ── serve ─────────────────────────────────────────────────────────────────────

def _serve(args):
    try:
        import uvicorn
    except ImportError:
        print("Install server dependencies: pip install 'ragpipe[server]'")
        sys.exit(1)

    pipeline = None

    if args.config:
        from ragpipe.config import PipelineConfig
        pipeline = PipelineConfig.from_yaml(args.config).build()

    from ragpipe.server.app import create_app
    app = create_app(pipeline=pipeline, api_key=args.api_key)

    print(f"Starting ragpipe server on {args.host}:{args.port}")
    if not pipeline:
        print("No pipeline config provided — set one via POST /ingest or restart with --config")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


# ── version ───────────────────────────────────────────────────────────────────

def _version(args):
    from ragpipe import __version__
    print(f"ragpipe v{__version__}")


if __name__ == "__main__":
    main()
