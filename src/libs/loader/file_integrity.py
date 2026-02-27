"""File integrity utilities for ingestion pipeline.

This module provides helpers to compute file hashes and decide whether
an input file should be skipped based on a simple local success cache.

The cache backend is intentionally minimal and file-based for now
(see DEV_SPEC C2). It can be swapped out later without changing the
public function signatures.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
from typing import Iterable, Set


_CACHE_ENV_VAR = "MODULAR_RAG_FILE_INTEGRITY_CACHE"
_DEFAULT_CACHE_PATH = "data/file_integrity_cache.json"


def _get_cache_path() -> Path:
    """Return the path to the cache file.

    The location can be overridden via the ``MODULAR_RAG_FILE_INTEGRITY_CACHE``
    environment variable. Otherwise it defaults to ``data/file_integrity_cache.json``
    relative to the project root.
    """

    override = os.getenv(_CACHE_ENV_VAR)
    if override:
        return Path(override)
    return Path(_DEFAULT_CACHE_PATH)


def compute_sha256(path: str | Path) -> str:
    """Compute the SHA256 hash of a file.

    Args:
        path: Path to the file on disk.

    Returns:
        Hex-encoded SHA256 digest of the file contents.
    """

    file_path = Path(path)
    hasher = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def _normalise_cache_data(data: object) -> Set[str]:
    """Normalise raw JSON data into a set of hashes.

    The cache file may store either a list of hashes or a mapping
    from hash to a truthy value. Any other structure is treated as
    empty.
    """

    if isinstance(data, list):
        return {str(item) for item in data}

    if isinstance(data, dict):  # type: ignore[redundant-cast]
        return {str(key) for key, value in data.items() if value}

    return set()


def _load_cache() -> Set[str]:
    """Load the success cache from disk.

    Returns:
        A set of file hash strings that have been marked as successful.
    """

    cache_path = _get_cache_path()
    if not cache_path.exists():
        return set()

    try:
        raw = cache_path.read_text(encoding="utf-8")
        if not raw.strip():
            return set()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        # Treat any I/O or decoding error as an empty cache; the
        # ingestion pipeline should never fail just because the
        # cache is corrupt.
        return set()

    return _normalise_cache_data(data)


def _save_cache(hashes: Iterable[str]) -> None:
    """Persist the given set of hashes to disk.

    The file is written atomically by first ensuring the parent
    directory exists and then overwriting the existing cache file.
    """

    cache_path = _get_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    unique_sorted = sorted({str(h) for h in hashes})
    cache_path.write_text(json.dumps(unique_sorted, ensure_ascii=False) + "\n", encoding="utf-8")


def should_skip(file_hash: str) -> bool:
    """Return True if a file with this hash should be skipped.

    A file is considered skippable if its hash has previously been
    recorded via :func:`mark_success`.
    """

    cache = _load_cache()
    return file_hash in cache


def mark_success(file_hash: str) -> None:
    """Record that processing a file with this hash succeeded.

    Subsequent calls to :func:`should_skip` with the same hash will
    return ``True``.
    """

    cache = _load_cache()
    if file_hash in cache:
        return

    cache.add(file_hash)
    _save_cache(cache)
