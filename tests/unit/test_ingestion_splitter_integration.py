"""Integration tests between ingestion pipeline and splitter layer.

These tests validate Stage C4 requirements:
- IngestionPipeline integrates with ``libs.splitter`` via SplitterFactory.
- Changing ingestion.chunk_size in settings affects the produced chunk
  lengths from the pipeline.
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


class DummySplitter(BaseSplitter):
    """Deterministic splitter used for testing.

    It splits text into fixed-size chunks based on the ``chunk_size``
    kwarg passed to ``split_text`` so tests can assert on the
    relationship between configuration and chunk lengths.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        size = int(kwargs.get("chunk_size", len(text))) or len(text)
        chunks = [text[i : i + size] for i in range(0, len(text), size)]
        self.validate_chunks(chunks)
        self.calls.append({"text": text, "trace": trace, "kwargs": kwargs, "chunks": chunks})
        return chunks


def _make_settings(chunk_size: int, chunk_overlap: int = 0) -> MagicMock:
    """Create a lightweight settings-like object for tests."""

    ingestion = MagicMock()
    ingestion.splitter = "dummy"
    ingestion.chunk_size = chunk_size
    ingestion.chunk_overlap = chunk_overlap

    settings = MagicMock()
    settings.ingestion = ingestion
    return settings


def test_pipeline_respects_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing chunk_size in settings should affect chunk lengths."""

    dummy_splitter = DummySplitter()

    def fake_create(settings: Any, **override_kwargs: Any) -> DummySplitter:
        # Factory must receive the same values as in settings.ingestion
        ingestion = settings.ingestion
        assert override_kwargs["chunk_size"] == ingestion.chunk_size
        assert override_kwargs["chunk_overlap"] == ingestion.chunk_overlap
        return dummy_splitter

    monkeypatch.setattr(SplitterFactory, "create", staticmethod(fake_create))

    text = "x" * 100

    small_settings = _make_settings(chunk_size=10, chunk_overlap=0)
    large_settings = _make_settings(chunk_size=25, chunk_overlap=0)

    small_pipeline = IngestionPipeline(small_settings)
    large_pipeline = IngestionPipeline(large_settings)

    small_chunks = small_pipeline.split_text(text)
    large_chunks = large_pipeline.split_text(text)

    max_small = max(len(c) for c in small_chunks)
    max_large = max(len(c) for c in large_chunks)

    assert max_small <= 10
    assert max_large <= 25
    # Larger configured chunk_size should lead to longer chunks overall
    assert max_large > max_small


def test_pipeline_requires_ingestion_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline should fail fast when ingestion settings are missing."""

    settings = MagicMock()
    settings.ingestion = None

    pipeline = IngestionPipeline(settings)

    with pytest.raises(ValueError, match="settings.ingestion"):
        pipeline.split_text("hello")
