"""CSV/Excel loader — converts tabular data to Document objects."""

from __future__ import annotations

from typing import Any

from ragpipe.core import Document


class CSVLoader:
    """Load CSV or Excel files as Documents.

    Each row becomes a text chunk with column headers as context,
    or the entire file can be loaded as a single document.

    Usage:
        loader = CSVLoader()
        docs = loader.load("data.csv")                   # one doc per row
        doc = loader.load("data.csv", one_document=True)  # single doc
        docs = loader.load("report.xlsx", sheet_name=0)   # Excel
    """

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def load(
        self,
        path: str,
        one_document: bool = False,
        sheet_name: Any = 0,
    ) -> list[Document] | Document:
        """Load a CSV or Excel file.

        Args:
            path: Path to .csv, .tsv, .xlsx, or .xls file.
            one_document: If True, return a single Document with all rows.
            sheet_name: For Excel files, which sheet to read (index or name).
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Install pandas: pip install pandas openpyxl")

        ext = path.rsplit(".", 1)[-1].lower()

        if ext in ("xlsx", "xls"):
            df = pd.read_excel(path, sheet_name=sheet_name)
        elif ext == "tsv":
            df = pd.read_csv(path, sep="\t", encoding=self.encoding)
        else:
            df = pd.read_csv(path, encoding=self.encoding)

        columns = list(df.columns)

        if one_document:
            lines = []
            for _, row in df.iterrows():
                parts = [f"{col}: {row[col]}" for col in columns if pd.notna(row[col])]
                lines.append(" | ".join(parts))
            content = "\n".join(lines)
            return Document(
                content=content,
                metadata={"source": path, "loader": "csv", "rows": len(df), "columns": columns},
            )

        docs = []
        for idx, row in df.iterrows():
            parts = [f"{col}: {row[col]}" for col in columns if pd.notna(row[col])]
            content = " | ".join(parts)
            docs.append(Document(
                content=content,
                metadata={
                    "source": path,
                    "loader": "csv",
                    "row_index": int(idx),
                    "columns": columns,
                },
            ))
        return docs
