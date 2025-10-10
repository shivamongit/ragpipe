"""Directory loader — recursively loads all supported files."""

from __future__ import annotations

from pathlib import Path

from ragpipe.core import Document
from ragpipe.loaders.text import TextLoader
from ragpipe.loaders.pdf import PDFLoader
from ragpipe.loaders.docx import DocxLoader


class DirectoryLoader:
    """Recursively load all supported files from a directory.

    Supported formats: .txt, .md, .markdown, .rst, .pdf, .docx
    Skips hidden files and directories.
    """

    def __init__(self, glob_pattern: str = "**/*"):
        self._glob = glob_pattern
        self._text_loader = TextLoader()

    def load(self, directory: str | Path) -> list[Document]:
        d = Path(directory)
        if not d.is_dir():
            raise ValueError(f"Not a directory: {d}")

        documents: list[Document] = []
        for path in sorted(d.glob(self._glob)):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.parts):
                continue

            ext = path.suffix.lower()
            try:
                if ext in TextLoader.EXTENSIONS:
                    documents.append(self._text_loader.load(path))
                elif ext == ".pdf":
                    documents.append(PDFLoader().load(path))
                elif ext == ".docx":
                    documents.append(DocxLoader().load(path))
            except Exception:
                continue

        return documents
