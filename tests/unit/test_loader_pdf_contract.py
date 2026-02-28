"""Contract tests for the PDF loader.

These tests validate the minimal behavior required by DEV_SPEC C3:
- ``PdfLoader.load`` can turn a sample PDF file into a ``Document``.
- The resulting ``Document.metadata`` contains at least ``source_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.models import Document
from src.libs.loader.pdf_loader import PdfLoader


def test_pdf_loader_loads_sample_pdf(sample_documents_dir: Path) -> None:
    """PdfLoader should produce a Document with source_path metadata."""

    pdf_path = sample_documents_dir / "sample.pdf"
    assert pdf_path.is_file(), "Expected sample.pdf to exist in sample_documents_dir"

    loader = PdfLoader()
    doc = loader.load(pdf_path)

    assert isinstance(doc, Document)
    assert doc.id == pdf_path.stem
    assert isinstance(doc.text, str)
    assert doc.text.strip() != ""
    assert doc.metadata.get("source_path") == str(pdf_path)


def test_pdf_loader_accepts_string_paths(sample_documents_dir: Path) -> None:
    """Loader should also accept string paths in addition to Path objects."""

    pdf_path = sample_documents_dir / "sample.pdf"
    loader = PdfLoader()

    doc = loader.load(str(pdf_path))

    assert isinstance(doc, Document)
    assert doc.metadata.get("source_path") == str(pdf_path)


def test_pdf_loader_raises_for_missing_file(sample_documents_dir: Path) -> None:
    """Loading a non-existent file should raise FileNotFoundError."""

    missing_path = sample_documents_dir / "does_not_exist.pdf"
    loader = PdfLoader()

    with pytest.raises(FileNotFoundError):
        loader.load(missing_path)
