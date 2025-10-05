"""Plain text and markdown file loader."""

from __future__ import annotations

from pathlib import Path

from ragpipe.core import Document


class TextLoader:
    """Load .txt and .md files into Document objects."""

    EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}

    def load(self, path: str | Path) -> Document:
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        return Document(
            content=content,
            metadata={"source": str(p), "filename": p.name, "type": p.suffix},
        )

    def load_many(self, paths: list[str | Path]) -> list[Document]:
        return [self.load(p) for p in paths]
