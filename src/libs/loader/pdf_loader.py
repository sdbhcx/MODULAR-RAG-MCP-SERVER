"""Minimal PDF loader implementation.

This is a Stage C3 shell implementation that satisfies the loader
contract for PDF files. It focuses on producing a ``Document`` with
basic metadata rather than full-fidelity PDF parsing.

In later stages the internal implementation can be swapped out for a
MarkItDown-based converter without changing the public interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.ingestion.models import Document
from src.libs.loader.base_loader import BaseLoader, PathLike


class PdfLoader(BaseLoader):
    """Loader for PDF documents.

    Current behavior is intentionally minimal:
    - Reads the underlying file as UTF-8 text (falling back to
      ``errors="ignore"`` for non-text bytes).
    - Uses the file stem as the ``Document.id``.
    - Always sets ``metadata["source_path"]`` to the absolute path
      string of the source file.
    """

    def load(
        self,
        path: PathLike,
        trace: Optional[Any] = None,
        **_: Any,
    ) -> Document:
        """Load a PDF file into a :class:`Document`.

        Args:
            path: Path to the PDF file on disk.
            trace: Optional TraceContext (unused for now).

        Returns:
            A :class:`Document` instance with text content and metadata.
        """

        file_path = self.validate_path(path)

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for binary-like content; this keeps the loader
            # robust even if the file is not a pure text PDF.
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        metadata = {"source_path": str(file_path)}
        return Document(id=file_path.stem, text=text, metadata=metadata)
