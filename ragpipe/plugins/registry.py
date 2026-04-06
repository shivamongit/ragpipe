"""Plugin registry — discover, register, and load ragpipe components.

Supports three discovery mechanisms:
1. Explicit registration via register()
2. Entry points (setuptools) for installed packages
3. Directory scanning for local plugin files

Usage:
    from ragpipe.plugins import PluginRegistry

    registry = PluginRegistry()

    # Explicit registration
    registry.register("chunker", "my_chunker", MyChunkerClass)

    # Discover via entry points (pip-installed plugins)
    registry.discover_entry_points()

    # Use a registered component
    chunker_cls = registry.get("chunker", "my_chunker")
    chunker = chunker_cls(chunk_size=512)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Type

logger = logging.getLogger(__name__)

# Plugin categories
CATEGORIES = (
    "chunker",
    "embedder",
    "retriever",
    "generator",
    "reranker",
    "loader",
    "agent",
    "guardrail",
    "evaluator",
)

ENTRY_POINT_GROUP = "ragpipe.plugins"


@dataclass
class PluginInfo:
    """Metadata about a registered plugin."""
    category: str
    name: str
    cls: Type | None = None
    module_path: str = ""
    description: str = ""
    version: str = ""
    author: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "class": f"{self.cls.__module__}.{self.cls.__name__}" if self.cls else "",
            "description": self.description,
            "version": self.version,
            "author": self.author,
        }


class PluginRegistry:
    """Central registry for ragpipe components.

    Provides a unified way to register, discover, and instantiate
    chunkers, embedders, retrievers, generators, and other components.
    """

    def __init__(self):
        self._plugins: dict[str, dict[str, PluginInfo]] = {
            cat: {} for cat in CATEGORIES
        }
        self._initialized = False

    def register(
        self,
        category: str,
        name: str,
        cls: Type,
        description: str = "",
        version: str = "",
        author: str = "",
        **metadata: Any,
    ) -> PluginInfo:
        """Register a component class under a category and name.

        Args:
            category: One of: chunker, embedder, retriever, generator, etc.
            name: Unique name within the category (e.g., "token", "ollama").
            cls: The component class.
            description: Optional description.
            version: Optional version string.
            author: Optional author.

        Returns:
            PluginInfo for the registered plugin.
        """
        if category not in self._plugins:
            self._plugins[category] = {}

        info = PluginInfo(
            category=category,
            name=name,
            cls=cls,
            module_path=f"{cls.__module__}.{cls.__name__}",
            description=description or cls.__doc__ or "",
            version=version,
            author=author,
            metadata=metadata,
        )
        self._plugins[category][name] = info
        logger.debug("Registered plugin: %s/%s -> %s", category, name, cls)
        return info

    def get(self, category: str, name: str) -> Type | None:
        """Get a registered component class by category and name."""
        info = self._plugins.get(category, {}).get(name)
        return info.cls if info else None

    def get_info(self, category: str, name: str) -> PluginInfo | None:
        """Get full plugin info by category and name."""
        return self._plugins.get(category, {}).get(name)

    def list_plugins(self, category: str | None = None) -> list[PluginInfo]:
        """List all registered plugins, optionally filtered by category."""
        if category:
            return list(self._plugins.get(category, {}).values())
        return [
            info
            for cat_plugins in self._plugins.values()
            for info in cat_plugins.values()
        ]

    def list_categories(self) -> list[str]:
        """List all categories that have at least one registered plugin."""
        return [cat for cat, plugins in self._plugins.items() if plugins]

    def discover_entry_points(self) -> int:
        """Discover and register plugins from installed packages via entry points.

        Entry points should be defined in pyproject.toml:
            [project.entry-points."ragpipe.plugins"]
            my_chunker = "my_package.chunkers:MyChunker"

        Or in setup.py:
            entry_points={"ragpipe.plugins": ["my_chunker = my_package.chunkers:MyChunker"]}

        Returns:
            Number of plugins discovered.
        """
        count = 0
        try:
            if hasattr(importlib.metadata, 'entry_points'):
                eps = importlib.metadata.entry_points()
                # Python 3.12+ returns a SelectableGroups or dict
                if isinstance(eps, dict):
                    group = eps.get(ENTRY_POINT_GROUP, [])
                else:
                    group = eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, 'select') else []

                for ep in group:
                    try:
                        cls = ep.load()
                        # Infer category from class inheritance or name
                        category = self._infer_category(cls, ep.name)
                        self.register(category, ep.name, cls)
                        count += 1
                    except Exception as e:
                        logger.warning("Failed to load plugin %s: %s", ep.name, e)
        except Exception as e:
            logger.debug("Entry point discovery not available: %s", e)

        return count

    def discover_module(self, module_path: str) -> int:
        """Import a module and register any ragpipe components found in it.

        Args:
            module_path: Dotted module path (e.g., "my_plugins.chunkers").

        Returns:
            Number of plugins discovered.
        """
        count = 0
        try:
            mod = importlib.import_module(module_path)
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type):
                    category = self._infer_category(attr, attr_name)
                    if category:
                        name = attr_name.lower().replace("chunker", "").replace("embedder", "")
                        name = name or attr_name.lower()
                        self.register(category, name, attr)
                        count += 1
        except ImportError as e:
            logger.warning("Could not import module %s: %s", module_path, e)
        return count

    def _infer_category(self, cls: Type, name: str = "") -> str:
        """Infer the plugin category from class hierarchy or naming convention."""
        class_name = cls.__name__.lower()
        name_lower = name.lower()

        # Check class hierarchy
        for parent in cls.__mro__:
            parent_name = parent.__name__.lower()
            if "chunker" in parent_name:
                return "chunker"
            if "embedder" in parent_name:
                return "embedder"
            if "retriever" in parent_name:
                return "retriever"
            if "generator" in parent_name:
                return "generator"
            if "reranker" in parent_name:
                return "reranker"
            if "loader" in parent_name:
                return "loader"

        # Check class name / registration name
        for cat in CATEGORIES:
            if cat in class_name or cat in name_lower:
                return cat

        return ""

    def register_builtins(self) -> None:
        """Register all built-in ragpipe components."""
        if self._initialized:
            return
        self._initialized = True

        # Chunkers
        try:
            from ragpipe.chunkers.token import TokenChunker
            self.register("chunker", "token", TokenChunker)
        except ImportError:
            pass
        try:
            from ragpipe.chunkers.recursive import RecursiveChunker
            self.register("chunker", "recursive", RecursiveChunker)
        except ImportError:
            pass

        # Embedders
        try:
            from ragpipe.embedders.ollama import OllamaEmbedder
            self.register("embedder", "ollama", OllamaEmbedder)
        except ImportError:
            pass

        # Retrievers
        try:
            from ragpipe.retrievers.numpy_retriever import NumpyRetriever
            self.register("retriever", "numpy", NumpyRetriever)
        except ImportError:
            pass
        try:
            from ragpipe.retrievers.faiss_retriever import FaissRetriever
            self.register("retriever", "faiss", FaissRetriever)
        except ImportError:
            pass
        try:
            from ragpipe.retrievers.bm25_retriever import BM25Retriever
            self.register("retriever", "bm25", BM25Retriever)
        except ImportError:
            pass

        # Generators
        try:
            from ragpipe.generators.ollama_gen import OllamaGenerator
            self.register("generator", "ollama", OllamaGenerator)
        except ImportError:
            pass

    def create(self, category: str, name: str, **kwargs: Any) -> Any:
        """Create an instance of a registered plugin.

        Args:
            category: Plugin category.
            name: Plugin name.
            **kwargs: Arguments to pass to the constructor.

        Returns:
            Instance of the plugin class.

        Raises:
            KeyError: If plugin not found.
        """
        cls = self.get(category, name)
        if cls is None:
            available = list(self._plugins.get(category, {}).keys())
            raise KeyError(
                f"Plugin '{name}' not found in category '{category}'. "
                f"Available: {available}"
            )
        return cls(**kwargs)

    def summary(self) -> str:
        """Human-readable summary of all registered plugins."""
        lines = ["Plugin Registry:"]
        for cat in CATEGORIES:
            plugins = self._plugins.get(cat, {})
            if plugins:
                lines.append(f"  {cat} ({len(plugins)}):")
                for name, info in plugins.items():
                    lines.append(f"    - {name}: {info.module_path}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """JSON-serializable registry."""
        return {
            cat: [info.to_dict() for info in plugins.values()]
            for cat, plugins in self._plugins.items()
            if plugins
        }


# Global singleton registry
_global_registry: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    """Get or create the global plugin registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
        _global_registry.register_builtins()
    return _global_registry
