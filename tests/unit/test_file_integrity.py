"""Unit tests for file integrity utilities."""

from __future__ import annotations

from pathlib import Path

import os

import pytest

from src.libs.loader.file_integrity import compute_sha256, should_skip, mark_success


_CACHE_ENV_VAR = "MODULAR_RAG_FILE_INTEGRITY_CACHE"


def test_compute_sha256_is_deterministic(tmp_path: Path) -> None:
    """Hashing the same file multiple times should yield the same value."""

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world", encoding="utf-8")

    h1 = compute_sha256(file_path)
    h2 = compute_sha256(str(file_path))  # also accept str paths

    assert h1 == h2
    # SHA256 hex digest should be 64 characters long
    assert len(h1) == 64


def test_mark_success_and_should_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Files marked as successful should be skipped on subsequent runs."""

    cache_path = tmp_path / "cache.json"
    monkeypatch.setenv(_CACHE_ENV_VAR, str(cache_path))

    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"some binary data")

    file_hash = compute_sha256(file_path)

    # Initially the file should not be skipped
    assert not should_skip(file_hash)

    # After marking success, should_skip must return True
    mark_success(file_hash)
    assert should_skip(file_hash)

    # Marking success again should be idempotent and still return True
    mark_success(file_hash)
    assert should_skip(file_hash)

    # A different hash should not be skipped
    other_path = tmp_path / "other.bin"
    other_path.write_bytes(b"other data")
    other_hash = compute_sha256(other_path)

    assert not should_skip(other_hash)
