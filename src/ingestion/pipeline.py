"""Ingestion pipeline components.

Stage C4 focuses on integrating the splitter layer into a minimal
pipeline wrapper so that splitter configuration (especially
``chunk_size`` and ``chunk_overlap``) flows correctly from settings
into the underlying splitter implementation.
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.core.settings import Settings
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


class IngestionPipeline:
    """Minimal ingestion pipeline for text splitting.

    For Stage C4 this pipeline only wires configuration from
    :class:`Settings` into the splitter factory and exposes a thin
    ``split_text`` wrapper around the configured splitter.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _create_splitter(self) -> BaseSplitter:
        """Create a splitter instance based on ingestion settings.

        Returns:
            A concrete splitter implementation created via
            :class:`SplitterFactory`.

        Raises:
            ValueError: If ingestion settings are missing.
        """

        ingestion = getattr(self._settings, "ingestion", None)
        if ingestion is None:
            raise ValueError("settings.ingestion must be configured for ingestion pipeline")

        return SplitterFactory.create(
            self._settings,
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
        )

    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        """Split text into chunks using the configured splitter.

        Args:
            text: Input text to split.
            trace: Optional TraceContext for observability (reserved
                for later stages).

        Returns:
            List of text chunks produced by the splitter.
        """

        ingestion = getattr(self._settings, "ingestion", None)
        if ingestion is None:
            raise ValueError("settings.ingestion must be configured for ingestion pipeline")

        splitter = self._create_splitter()
        return splitter.split_text(
            text,
            trace=trace,
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
        )
