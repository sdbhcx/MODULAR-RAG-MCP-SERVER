"""Abstract base class for document loaders.

Loader components are responsible for turning raw files on disk into
canonical ``Document`` objects used by the ingestion pipeline.

Design principles:
- Pluggable: concrete loaders implement a common ``BaseLoader``
  interface and can be swapped without changing callers.
- Config-Driven: which loader to use will later be decided by
  pipeline configuration (e.g. by file extension).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from src.ingestion.models import Document


PathLike = Union[str, Path]


class BaseLoader(ABC):
    """Abstract base class for document loaders.

    Subclasses must implement :meth:`load` to read a file from disk and
    produce a :class:`Document` instance with appropriate metadata.
    """

    @abstractmethod
    def load(
        self,
        path: PathLike,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Document:
        """Load a document from the given path.

        Args:
            path: Path to the source file on disk.
            trace: Optional TraceContext for observability (reserved for
                later stages).
            **kwargs: Loader-specific options.

        Returns:
            A :class:`Document` representing the loaded file.
        """
        raise NotImplementedError

    def validate_path(self, path: PathLike) -> Path:
        """Validate that the given path points to an existing file.

        Args:
            path: String or :class:`Path` to validate.

        Returns:
            A normalised :class:`Path` object.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the path is not a file.
        """

        file_path = path if isinstance(path, Path) else Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Expected a file path, got: {file_path}")

        return file_path
