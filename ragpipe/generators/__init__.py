from ragpipe.generators.base import BaseGenerator, GenerationOutput
from ragpipe.generators.ollama_gen import OllamaGenerator

__all__ = [
    "BaseGenerator",
    "GenerationOutput",
    "OllamaGenerator",
]

# Optional generators — imported only if their dependencies are installed
def __getattr__(name):
    if name == "OpenAIGenerator":
        from ragpipe.generators.openai_gen import OpenAIGenerator
        return OpenAIGenerator
    if name == "AnthropicGenerator":
        from ragpipe.generators.anthropic_gen import AnthropicGenerator
        return AnthropicGenerator
    if name == "LiteLLMGenerator":
        from ragpipe.generators.litellm_gen import LiteLLMGenerator
        return LiteLLMGenerator
    raise AttributeError(f"module 'ragpipe.generators' has no attribute {name!r}")
