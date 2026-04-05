"""HTML/Web loader — extract text from HTML files or URLs."""

from __future__ import annotations

from ragpipe.core import Document


class HTMLLoader:
    """Load HTML files or web URLs as Documents.

    Uses BeautifulSoup for parsing. Extracts visible text,
    strips scripts/styles, preserves paragraph structure.

    Usage:
        loader = HTMLLoader()
        doc = loader.load("page.html")
        doc = loader.load_url("https://example.com/article")
    """

    def __init__(self, parser: str = "html.parser"):
        self.parser = parser

    def _extract_text(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

        soup = BeautifulSoup(html, self.parser)

        # Remove scripts, styles, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract text with paragraph separation
        text = soup.get_text(separator="\n", strip=True)

        # Collapse multiple blank lines
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n\n".join(lines)

    def _extract_title(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return ""

        soup = BeautifulSoup(html, self.parser)
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else ""

    def load(self, path: str) -> Document:
        """Load an HTML file from disk."""
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        content = self._extract_text(html)
        title = self._extract_title(html)

        return Document(
            content=content,
            metadata={
                "source": path,
                "loader": "html",
                "title": title,
            },
        )

    def load_url(self, url: str, timeout: float = 30.0) -> Document:
        """Fetch and parse a web URL."""
        import httpx

        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        content = self._extract_text(html)
        title = self._extract_title(html)

        return Document(
            content=content,
            metadata={
                "source": url,
                "loader": "web",
                "title": title,
                "status_code": resp.status_code,
            },
        )

    async def aload_url(self, url: str, timeout: float = 30.0) -> Document:
        """Async fetch and parse a web URL."""
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text

        content = self._extract_text(html)
        title = self._extract_title(html)

        return Document(
            content=content,
            metadata={
                "source": url,
                "loader": "web",
                "title": title,
                "status_code": resp.status_code,
            },
        )
