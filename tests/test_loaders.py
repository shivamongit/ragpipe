"""Tests for document loaders."""

import tempfile
from pathlib import Path

from ragpipe.loaders.text import TextLoader
from ragpipe.loaders.directory import DirectoryLoader


def test_text_loader():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world from a text file.")
        f.flush()
        loader = TextLoader()
        doc = loader.load(f.name)
        assert doc.content == "Hello world from a text file."
        assert doc.metadata["type"] == ".txt"


def test_text_loader_markdown():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Title\n\nSome markdown content.")
        f.flush()
        loader = TextLoader()
        doc = loader.load(f.name)
        assert "# Title" in doc.content


def test_directory_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "file1.txt").write_text("Content one.")
        (Path(tmpdir) / "file2.md").write_text("Content two.")
        (Path(tmpdir) / "ignore.py").write_text("not loaded")

        loader = DirectoryLoader()
        docs = loader.load(tmpdir)
        assert len(docs) == 2
        contents = {d.content for d in docs}
        assert "Content one." in contents
        assert "Content two." in contents
