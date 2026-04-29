from ragpipe.generators.base import BaseGenerator, GenerationOutput
from ragpipe.generators.ollama_gen import OllamaGenerator
from ragpipe.generators.registry import (
    PROVIDERS,
    ModelInfo,
    ProviderInfo,
    build_generator,
    find_model,
    list_providers,
)

__all__ = [
    "BaseGenerator",
    "GenerationOutput",
    "OllamaGenerator",
    # Registry
    "PROVIDERS",
    "ModelInfo",
    "ProviderInfo",
    "build_generator",
    "find_model",
    "list_providers",
]


# Optional generators — imported only if their dependencies are installed
def __getattr__(name):
    if name == "OpenAIGenerator":
        from ragpipe.generators.openai_gen import OpenAIGenerator
        return OpenAIGenerator
    if name == "AnthropicGenerator":
        from ragpipe.generators.anthropic_gen import AnthropicGenerator
        return AnthropicGenerator
    if name == "GeminiGenerator":
        from ragpipe.generators.gemini_gen import GeminiGenerator
        return GeminiGenerator
    if name == "GroqGenerator":
        from ragpipe.generators.groq_gen import GroqGenerator
        return GroqGenerator
    if name == "CohereGenerator":
        from ragpipe.generators.cohere_gen import CohereGenerator
        return CohereGenerator
    if name == "MistralGenerator":
        from ragpipe.generators.mistral_gen import MistralGenerator
        return MistralGenerator
    if name == "LiteLLMGenerator":
        from ragpipe.generators.litellm_gen import LiteLLMGenerator
        return LiteLLMGenerator
    raise AttributeError(f"module 'ragpipe.generators' has no attribute {name!r}")
