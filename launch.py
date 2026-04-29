#!/usr/bin/env python3
"""ragpipe launcher — start a fully-configured server with one command.

Auto-detects available LLM providers from environment variables and falls back
to Ollama (local) if available. Supports overriding via CLI args.

Usage:
    python launch.py                          # auto-detect best available
    python launch.py --provider openai        # force OpenAI as default
    python launch.py --model gpt-5-mini       # specific default model
    python launch.py --port 8000              # custom port
    python launch.py --db ragpipe.db          # SQLite path for conversations
"""

from __future__ import annotations

import argparse
import os
import sys


PROVIDER_PRIORITY = ["anthropic", "openai", "google", "groq", "cohere", "mistral", "ollama"]
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-mini",
    "google": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "cohere": "command-r",
    "mistral": "mistral-small-2503",
    "ollama": "gemma3:4b",
}


def has_env_key(provider: str) -> bool:
    env_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "cohere": "COHERE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }.get(provider)
    if not env_var:
        return True  # ollama needs no key
    return bool(os.environ.get(env_var))


def has_package(provider: str) -> bool:
    import importlib.util
    pkg = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google.genai",
        "groq": "openai",
        "cohere": "cohere",
        "mistral": "mistralai",
        "ollama": None,
    }.get(provider)
    if pkg is None:
        return True
    return importlib.util.find_spec(pkg.split(".")[0]) is not None


def is_ollama_running() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.0) as r:
            return r.status == 200
    except Exception:
        return False


def auto_detect_provider() -> tuple[str, str]:
    """Return (provider, model) for the highest-priority available combination."""
    for provider in PROVIDER_PRIORITY:
        if not has_package(provider):
            continue
        if provider == "ollama":
            if is_ollama_running():
                return "ollama", DEFAULT_MODELS["ollama"]
            continue
        if has_env_key(provider):
            return provider, DEFAULT_MODELS[provider]
    raise RuntimeError(
        "No LLM provider available. Set one of:\n"
        "  - OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY,\n"
        "    COHERE_API_KEY, MISTRAL_API_KEY\n"
        "  - Or install and run Ollama (https://ollama.com)"
    )


def build_pipeline(provider: str, model: str):
    from ragpipe.core import Pipeline
    from ragpipe.chunkers import RecursiveChunker
    from ragpipe.retrievers import NumpyRetriever
    from ragpipe.generators.registry import build_generator

    # Embedder — prefer ollama if running, else sentence-transformers, else openai if available
    embedder = _build_embedder()

    print(f"  Embedder:  {type(embedder).__name__}")
    print(f"  Provider:  {provider}")
    print(f"  Model:     {model}")

    generator = build_generator(provider, model)

    return Pipeline(
        chunker=RecursiveChunker(chunk_size=512, overlap=64),
        embedder=embedder,
        retriever=NumpyRetriever(),
        generator=generator,
    )


def _build_embedder():
    """Pick the first available embedder in priority order."""
    # 1. Local Ollama (free + private)
    if is_ollama_running():
        try:
            from ragpipe.embedders.ollama import OllamaEmbedder
            return OllamaEmbedder(model="nomic-embed-text")
        except Exception:
            pass
    # 2. OpenAI embeddings
    if has_package("openai") and os.environ.get("OPENAI_API_KEY"):
        try:
            from ragpipe.embedders.openai import OpenAIEmbedder
            return OpenAIEmbedder(model="text-embedding-3-small")
        except Exception:
            pass
    # 3. Sentence-transformers (local, free, but slower first run)
    try:
        from ragpipe.embedders.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    except ImportError:
        pass
    raise RuntimeError(
        "No embedder available. Either:\n"
        "  - Run Ollama with `ollama pull nomic-embed-text`\n"
        "  - Set OPENAI_API_KEY\n"
        "  - Install: pip install 'ragpipe[sentence-transformers]'"
    )


def main():
    ap = argparse.ArgumentParser(description="Launch ragpipe server with auto-detected provider")
    ap.add_argument("--provider", help="Force provider (openai/anthropic/google/groq/cohere/mistral/ollama)")
    ap.add_argument("--model", help="Specific model id (defaults to provider's recommended model)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default="ragpipe.db", help="SQLite path for conversation history")
    ap.add_argument("--api-key", help="Optional X-API-Key for the server")
    args = ap.parse_args()

    print("\n┌─ ragpipe launcher ──────────────────────────────")
    print(f"│ Loading .env (if present)...")

    _load_dotenv()

    if args.provider:
        provider = args.provider
        model = args.model or DEFAULT_MODELS.get(provider, "")
        if not model:
            print(f"│ ✗ No default model known for provider '{provider}'. Pass --model.")
            sys.exit(1)
    else:
        try:
            provider, model = auto_detect_provider()
        except RuntimeError as e:
            print(f"│ ✗ {e}")
            sys.exit(1)
        if args.model:
            model = args.model

    print(f"│ Building pipeline...")
    pipeline = build_pipeline(provider, model)

    try:
        import uvicorn
        from ragpipe.server.app import create_app
    except ImportError:
        print("│ ✗ Install server deps: pip install 'ragpipe[server]'")
        sys.exit(1)

    app = create_app(pipeline=pipeline, api_key=args.api_key, db_path=args.db)

    print(f"│ DB:        {args.db}")
    print(f"│ Listening: http://{args.host}:{args.port}")
    print(f"│ UI:        http://localhost:3000  (run `npm run dev` in ragpipe-ui/)")
    print("└─────────────────────────────────────────────────\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _load_dotenv():
    """Tiny .env loader (no external dep). Reads from CWD/.env if present."""
    env_path = ".env"
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


if __name__ == "__main__":
    main()
