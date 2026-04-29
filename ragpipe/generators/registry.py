"""Unified registry of LLM providers and models for runtime model switching.

This is the single source of truth for what models are available, used by:
- The /providers and /models REST endpoints
- The UI model picker
- Runtime generator instantiation in the server
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelInfo:
    """Metadata for a single model offered by a provider."""

    id: str  # e.g. "gpt-5-mini", "claude-sonnet-4-6"
    name: str  # human-readable label
    provider: str  # "openai", "anthropic", "google", "groq", "cohere", "mistral", "ollama"
    context_window: int = 8192
    input_cost_per_m: float = 0.0  # USD per million input tokens
    output_cost_per_m: float = 0.0  # USD per million output tokens
    streaming: bool = True
    description: str = ""
    tags: list[str] = field(default_factory=list)  # e.g. ["fast", "reasoning", "cheap"]


@dataclass
class ProviderInfo:
    """Metadata for an LLM provider."""

    id: str  # e.g. "openai"
    name: str  # human label e.g. "OpenAI"
    requires_api_key: bool = True
    api_key_env_var: str = ""
    docs_url: str = ""
    available: bool = False  # set at runtime based on installed packages + keys
    models: list[ModelInfo] = field(default_factory=list)


# Curated catalog of models, kept in sync with the generator implementations.
PROVIDERS: dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        id="openai",
        name="OpenAI",
        api_key_env_var="OPENAI_API_KEY",
        docs_url="https://platform.openai.com/docs/models",
        models=[
            ModelInfo("gpt-5.4", "GPT-5.4", "openai", 1_000_000, 2.50, 15.0, description="Flagship multimodal", tags=["flagship"]),
            ModelInfo("gpt-5.4-pro", "GPT-5.4 Pro", "openai", 1_000_000, 30.0, 180.0, description="Max reasoning", tags=["reasoning"]),
            ModelInfo("gpt-5", "GPT-5", "openai", 256_000, 1.25, 10.0, description="General purpose", tags=["balanced"]),
            ModelInfo("gpt-5-mini", "GPT-5 Mini", "openai", 256_000, 0.25, 2.0, description="Balanced cost/quality", tags=["balanced", "cheap"]),
            ModelInfo("gpt-5-nano", "GPT-5 Nano", "openai", 128_000, 0.05, 0.40, description="Cheapest", tags=["cheap"]),
            ModelInfo("gpt-4.1", "GPT-4.1", "openai", 128_000, 2.0, 8.0, description="Legacy flagship", tags=["legacy"]),
        ],
    ),
    "anthropic": ProviderInfo(
        id="anthropic",
        name="Anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        docs_url="https://docs.anthropic.com/en/docs/about-claude/models",
        models=[
            ModelInfo("claude-opus-4-6", "Claude Opus 4.6", "anthropic", 200_000, 5.0, 25.0, description="Flagship", tags=["flagship"]),
            ModelInfo("claude-sonnet-4-6", "Claude Sonnet 4.6", "anthropic", 200_000, 3.0, 15.0, description="Best value", tags=["balanced"]),
            ModelInfo("claude-haiku-4-5", "Claude Haiku 4.5", "anthropic", 200_000, 1.0, 5.0, description="Fast & cheap", tags=["fast", "cheap"]),
        ],
    ),
    "google": ProviderInfo(
        id="google",
        name="Google Gemini",
        api_key_env_var="GOOGLE_API_KEY",
        docs_url="https://ai.google.dev/gemini-api/docs/models",
        models=[
            ModelInfo("gemini-2.5-pro", "Gemini 2.5 Pro", "google", 2_000_000, 1.25, 10.0, description="Flagship, 2M ctx", tags=["flagship", "long-context"]),
            ModelInfo("gemini-2.5-flash", "Gemini 2.5 Flash", "google", 1_000_000, 0.30, 2.50, description="Fast & cheap", tags=["fast", "balanced"]),
            ModelInfo("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", "google", 1_000_000, 0.10, 0.40, description="Ultra cheap", tags=["cheap"]),
            ModelInfo("gemini-2.0-flash", "Gemini 2.0 Flash", "google", 1_000_000, 0.10, 0.40, description="Previous gen", tags=["legacy", "cheap"]),
        ],
    ),
    "groq": ProviderInfo(
        id="groq",
        name="Groq (LPU)",
        api_key_env_var="GROQ_API_KEY",
        docs_url="https://console.groq.com/docs/models",
        models=[
            ModelInfo("llama-3.3-70b-versatile", "Llama 3.3 70B", "groq", 128_000, 0.59, 0.79, description="Best balance, ~275 tok/s", tags=["fast", "balanced"]),
            ModelInfo("llama-3.1-8b-instant", "Llama 3.1 8B Instant", "groq", 128_000, 0.05, 0.08, description="Fastest, ~750 tok/s", tags=["fast", "cheap"]),
            ModelInfo("mixtral-8x7b-32768", "Mixtral 8x7B", "groq", 32_768, 0.24, 0.24, description="Long context", tags=["balanced"]),
            ModelInfo("qwen-3-32b", "Qwen 3 32B", "groq", 128_000, 0.29, 0.59, description="Strong reasoning", tags=["reasoning"]),
            ModelInfo("deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill 70B", "groq", 128_000, 0.75, 0.99, description="Reasoning model", tags=["reasoning"]),
            ModelInfo("kimi-k2", "Kimi K2", "groq", 128_000, 0.50, 0.50, description="Agentic", tags=["agentic"]),
        ],
    ),
    "cohere": ProviderInfo(
        id="cohere",
        name="Cohere",
        api_key_env_var="COHERE_API_KEY",
        docs_url="https://docs.cohere.com/docs/models",
        models=[
            ModelInfo("command-a-03-2025", "Command A", "cohere", 256_000, 2.50, 10.0, description="Flagship", tags=["flagship"]),
            ModelInfo("command-r-plus", "Command R+", "cohere", 128_000, 2.50, 10.0, description="Best for RAG", tags=["balanced", "rag"]),
            ModelInfo("command-r", "Command R", "cohere", 128_000, 0.50, 1.50, description="Balanced", tags=["balanced"]),
            ModelInfo("command-r7b", "Command R7B", "cohere", 128_000, 0.0375, 0.15, description="Tiny & cheap", tags=["cheap"]),
        ],
    ),
    "mistral": ProviderInfo(
        id="mistral",
        name="Mistral AI",
        api_key_env_var="MISTRAL_API_KEY",
        docs_url="https://docs.mistral.ai/getting-started/models/",
        models=[
            ModelInfo("mistral-large-2411", "Mistral Large", "mistral", 128_000, 2.0, 6.0, description="Flagship", tags=["flagship"]),
            ModelInfo("mistral-medium-2505", "Mistral Medium", "mistral", 128_000, 0.40, 2.0, description="Balanced", tags=["balanced"]),
            ModelInfo("mistral-small-2503", "Mistral Small", "mistral", 128_000, 0.10, 0.30, description="Efficient", tags=["cheap"]),
            ModelInfo("codestral-2501", "Codestral", "mistral", 256_000, 0.20, 0.60, description="Coding specialist", tags=["coding"]),
        ],
    ),
    "ollama": ProviderInfo(
        id="ollama",
        name="Ollama (Local)",
        requires_api_key=False,
        api_key_env_var="",
        docs_url="https://ollama.com/library",
        models=[
            ModelInfo("llama3.3:70b", "Llama 3.3 70B", "ollama", 128_000, 0.0, 0.0, description="Local, free", tags=["local", "free"]),
            ModelInfo("llama3.2:3b", "Llama 3.2 3B", "ollama", 128_000, 0.0, 0.0, description="Tiny local", tags=["local", "free", "fast"]),
            ModelInfo("gemma3:4b", "Gemma 3 4B", "ollama", 128_000, 0.0, 0.0, description="Local, fast", tags=["local", "free", "fast"]),
            ModelInfo("gemma3:27b", "Gemma 3 27B", "ollama", 128_000, 0.0, 0.0, description="Local, larger", tags=["local", "free"]),
            ModelInfo("qwen2.5:7b", "Qwen 2.5 7B", "ollama", 128_000, 0.0, 0.0, description="Local, balanced", tags=["local", "free"]),
            ModelInfo("phi4:latest", "Phi-4", "ollama", 16_000, 0.0, 0.0, description="MS small model", tags=["local", "free"]),
            ModelInfo("mistral:7b", "Mistral 7B", "ollama", 32_000, 0.0, 0.0, description="Local Mistral", tags=["local", "free"]),
            ModelInfo("deepseek-r1:7b", "DeepSeek R1 7B", "ollama", 128_000, 0.0, 0.0, description="Reasoning", tags=["local", "free", "reasoning"]),
        ],
    ),
}


def list_providers() -> list[ProviderInfo]:
    """Return all provider metadata, with `available` updated by env/install state."""
    import os
    import importlib.util

    # Map provider id -> python package import name to detect installation
    pkg_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google.genai",
        "groq": "openai",  # groq uses the openai-compatible client
        "cohere": "cohere",
        "mistral": "mistralai",
        "ollama": None,  # no package required
    }

    out = []
    for pid, prov in PROVIDERS.items():
        # Copy so we don't mutate the global
        p = ProviderInfo(
            id=prov.id,
            name=prov.name,
            requires_api_key=prov.requires_api_key,
            api_key_env_var=prov.api_key_env_var,
            docs_url=prov.docs_url,
            models=list(prov.models),
        )
        pkg = pkg_map.get(pid)
        installed = True if pkg is None else (importlib.util.find_spec(pkg.split(".")[0]) is not None)
        has_key = (not p.requires_api_key) or bool(os.environ.get(p.api_key_env_var))
        p.available = installed and has_key
        out.append(p)
    return out


def find_model(model_id: str, provider: Optional[str] = None) -> Optional[tuple[ProviderInfo, ModelInfo]]:
    """Look up a model by id, optionally constrained to a provider."""
    for prov in PROVIDERS.values():
        if provider and prov.id != provider:
            continue
        for m in prov.models:
            if m.id == model_id:
                return prov, m
    return None


def build_generator(provider: str, model: str, api_key: Optional[str] = None, **kwargs: Any):
    """Instantiate the appropriate generator for a (provider, model) pair.

    Args:
        provider: Provider id ("openai", "anthropic", "google", "groq", "cohere", "mistral", "ollama")
        model: Model id (e.g. "gpt-5-mini", "claude-sonnet-4-6")
        api_key: Optional API key override. If None, falls back to env var for the provider.
        **kwargs: Extra args forwarded to the generator constructor (temperature, max_tokens, ...)

    Returns:
        A configured BaseGenerator instance.

    Raises:
        ValueError: If provider is unknown.
        ImportError: If the provider's package is not installed.
    """
    if provider == "openai":
        from ragpipe.generators.openai_gen import OpenAIGenerator
        return OpenAIGenerator(model=model, api_key=api_key, **kwargs)
    if provider == "anthropic":
        from ragpipe.generators.anthropic_gen import AnthropicGenerator
        return AnthropicGenerator(model=model, api_key=api_key, **kwargs)
    if provider == "google":
        from ragpipe.generators.gemini_gen import GeminiGenerator
        return GeminiGenerator(model=model, api_key=api_key, **kwargs)
    if provider == "groq":
        from ragpipe.generators.groq_gen import GroqGenerator
        return GroqGenerator(model=model, api_key=api_key, **kwargs)
    if provider == "cohere":
        from ragpipe.generators.cohere_gen import CohereGenerator
        return CohereGenerator(model=model, api_key=api_key, **kwargs)
    if provider == "mistral":
        from ragpipe.generators.mistral_gen import MistralGenerator
        return MistralGenerator(model=model, api_key=api_key, **kwargs)
    if provider == "ollama":
        from ragpipe.generators.ollama_gen import OllamaGenerator
        return OllamaGenerator(model=model, **kwargs)
    raise ValueError(f"Unknown provider: {provider!r}")
