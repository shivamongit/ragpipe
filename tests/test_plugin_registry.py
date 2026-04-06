"""Tests for ragpipe.plugins — Plugin system."""

from ragpipe.plugins.registry import PluginRegistry, PluginInfo, CATEGORIES


class _MockChunker:
    """A mock chunker for testing."""
    pass


class _MockEmbedder:
    """A mock embedder for testing."""
    pass


# ── PluginInfo ────────────────────────────────────────────────────────────────

def test_plugin_info_to_dict():
    info = PluginInfo(category="chunker", name="mock", cls=_MockChunker, description="test")
    d = info.to_dict()
    assert d["category"] == "chunker"
    assert d["name"] == "mock"
    assert "_MockChunker" in d["class"]


# ── PluginRegistry basics ────────────────────────────────────────────────────

def test_registry_init():
    reg = PluginRegistry()
    assert reg.list_plugins() == []


def test_registry_register():
    reg = PluginRegistry()
    info = reg.register("chunker", "mock", _MockChunker, description="A mock chunker")
    assert info.category == "chunker"
    assert info.name == "mock"
    assert info.cls is _MockChunker


def test_registry_get():
    reg = PluginRegistry()
    reg.register("chunker", "mock", _MockChunker)
    cls = reg.get("chunker", "mock")
    assert cls is _MockChunker


def test_registry_get_not_found():
    reg = PluginRegistry()
    assert reg.get("chunker", "nonexistent") is None


def test_registry_get_info():
    reg = PluginRegistry()
    reg.register("chunker", "mock", _MockChunker, version="1.0")
    info = reg.get_info("chunker", "mock")
    assert info is not None
    assert info.version == "1.0"


def test_registry_list_plugins():
    reg = PluginRegistry()
    reg.register("chunker", "mock1", _MockChunker)
    reg.register("embedder", "mock2", _MockEmbedder)
    all_plugins = reg.list_plugins()
    assert len(all_plugins) == 2


def test_registry_list_plugins_by_category():
    reg = PluginRegistry()
    reg.register("chunker", "mock1", _MockChunker)
    reg.register("embedder", "mock2", _MockEmbedder)
    chunkers = reg.list_plugins("chunker")
    assert len(chunkers) == 1
    assert chunkers[0].name == "mock1"


def test_registry_list_categories():
    reg = PluginRegistry()
    reg.register("chunker", "mock", _MockChunker)
    cats = reg.list_categories()
    assert "chunker" in cats


# ── create ────────────────────────────────────────────────────────────────────

def test_registry_create():
    class Simple:
        def __init__(self, x=1):
            self.x = x

    reg = PluginRegistry()
    reg.register("chunker", "simple", Simple)
    instance = reg.create("chunker", "simple", x=42)
    assert instance.x == 42


def test_registry_create_not_found():
    reg = PluginRegistry()
    import pytest
    with pytest.raises(KeyError, match="not found"):
        reg.create("chunker", "nonexistent")


# ── Category inference ────────────────────────────────────────────────────────

def test_infer_category_from_name():
    reg = PluginRegistry()
    category = reg._infer_category(_MockChunker, "my_chunker")
    assert category == "chunker"


def test_infer_category_from_class_name():
    class MyRetriever:
        pass
    reg = PluginRegistry()
    category = reg._infer_category(MyRetriever)
    assert category == "retriever"


# ── Entry points (no actual entry points installed) ───────────────────────────

def test_discover_entry_points_no_crash():
    reg = PluginRegistry()
    count = reg.discover_entry_points()
    assert isinstance(count, int)


# ── Serialization ─────────────────────────────────────────────────────────────

def test_registry_summary():
    reg = PluginRegistry()
    reg.register("chunker", "mock", _MockChunker)
    s = reg.summary()
    assert "chunker" in s
    assert "mock" in s


def test_registry_to_dict():
    reg = PluginRegistry()
    reg.register("chunker", "mock", _MockChunker)
    d = reg.to_dict()
    assert "chunker" in d
    assert len(d["chunker"]) == 1


# ── Builtins ──────────────────────────────────────────────────────────────────

def test_register_builtins():
    reg = PluginRegistry()
    reg.register_builtins()
    # Should have at least some built-in chunkers and retrievers
    all_plugins = reg.list_plugins()
    assert len(all_plugins) >= 1


def test_register_builtins_idempotent():
    reg = PluginRegistry()
    reg.register_builtins()
    count1 = len(reg.list_plugins())
    reg.register_builtins()
    count2 = len(reg.list_plugins())
    assert count1 == count2
