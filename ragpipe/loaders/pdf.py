"""PDF file loader."""

from __future__ import annotations

from pathlib import Path

from ragpipe.core import Document


class PDFLoader:
    """Load PDF files into Document objects.

    Extracts text from all pages. Page metadata is preserved.
    """

    def load(self, path: str | Path) -> Document:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("Install PyPDF2: pip install 'ragpipe[pdf]'")

        p = Path(path)
        reader = PdfReader(str(p))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(text.strip())

        return Document(
            content="\n\n".join(pages),
            metadata={
                "source": str(p),
                "filename": p.name,
                "type": ".pdf",
                "page_count": len(reader.pages),
            },
        )

    def load_many(self, paths: list[str | Path]) -> list[Document]:
        return [self.load(p) for p in paths]
