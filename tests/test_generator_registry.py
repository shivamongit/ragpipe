"""Tests for the unified LLM provider/model registry."""

from __future__ import annotations

import pytest

from ragpipe.generators.registry import (
    PROVIDERS,
    ModelInfo,
    ProviderInfo,
    build_generator,
    find_model,
    list_providers,
)


def test_providers_dict_populated():
    assert "openai" in PROVIDERS
    assert "anthropic" in PROVIDERS
    assert "google" in PROVIDERS
    assert "groq" in PROVIDERS
    assert "cohere" in PROVIDERS
    assert "mistral" in PROVIDERS
    assert "ollama" in PROVIDERS


def test_each_provider_has_models():
    for pid, prov in PROVIDERS.items():
        assert len(prov.models) > 0, f"Provider {pid} has no models"
        for m in prov.models:
            assert m.id
            assert m.name
            assert m.provider == pid


def test_ollama_does_not_require_api_key():
    assert PROVIDERS["ollama"].requires_api_key is False


def test_openai_requires_api_key():
    assert PROVIDERS["openai"].requires_api_key is True
    assert PROVIDERS["openai"].api_key_env_var == "OPENAI_API_KEY"


def test_list_providers_returns_availability():
    provs = list_providers()
    assert len(provs) == len(PROVIDERS)
    for p in provs:
        assert isinstance(p.available, bool)


def test_find_model_existing():
    result = find_model("gpt-5-mini")
    assert result is not None
    prov, model = result
    assert prov.id == "openai"
    assert model.id == "gpt-5-mini"


def test_find_model_with_provider_filter():
    result = find_model("gpt-5-mini", provider="openai")
    assert result is not None
    result_wrong_prov = find_model("gpt-5-mini", provider="anthropic")
    assert result_wrong_prov is None


def test_find_model_unknown_returns_none():
    assert find_model("totally-fake-model-xyz") is None


def test_build_generator_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_generator("nonexistent", "some-model")


def test_model_info_dataclass_defaults():
    m = ModelInfo(id="x", name="X", provider="p")
    assert m.context_window == 8192
    assert m.input_cost_per_m == 0.0
    assert m.streaming is True
    assert m.tags == []


def test_provider_info_dataclass_defaults():
    p = ProviderInfo(id="x", name="X")
    assert p.requires_api_key is True
    assert p.available is False
    assert p.models == []


def test_models_have_consistent_provider_ids():
    """Each model's provider field must match the provider key it's listed under."""
    for pid, prov in PROVIDERS.items():
        for m in prov.models:
            assert m.provider == pid, (
                f"Model {m.id} listed under {pid} but has provider={m.provider}"
            )


def test_groq_uses_openai_pkg():
    """Groq generator uses the OpenAI client (OpenAI-compatible API)."""
    # Just verify the provider is registered; the import test is in build_generator
    assert "groq" in PROVIDERS
    assert PROVIDERS["groq"].api_key_env_var == "GROQ_API_KEY"
