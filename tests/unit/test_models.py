"""Unit tests for core data models."""

import pytest
from src.ingestion.models import Document, Chunk


def test_document_serialization():
    """Test Document serialization and deserialization."""
    doc = Document(
        id="doc_1",
        text="This is a test document.",
        metadata={"source": "test.pdf", "author": "Alice"}
    )
    
    doc_dict = doc.to_dict()
    assert doc_dict["id"] == "doc_1"
    assert doc_dict["text"] == "This is a test document."
    assert doc_dict["metadata"]["source"] == "test.pdf"
    
    doc_restored = Document.from_dict(doc_dict)
    assert doc_restored.id == doc.id
    assert doc_restored.text == doc.text
    assert doc_restored.metadata == doc.metadata


def test_document_default_metadata():
    """Test Document with default metadata."""
    doc = Document(id="doc_2", text="No metadata.")
    assert doc.metadata == {}
    
    doc_dict = doc.to_dict()
    doc_restored = Document.from_dict(doc_dict)
    assert doc_restored.metadata == {}


def test_chunk_serialization():
    """Test Chunk serialization and deserialization."""
    chunk = Chunk(
        id="chunk_1",
        text="This is a chunk.",
        metadata={"doc_id": "doc_1", "page": 1},
        start_offset=10,
        end_offset=26
    )
    
    chunk_dict = chunk.to_dict()
    assert chunk_dict["id"] == "chunk_1"
    assert chunk_dict["text"] == "This is a chunk."
    assert chunk_dict["metadata"]["doc_id"] == "doc_1"
    assert chunk_dict["start_offset"] == 10
    assert chunk_dict["end_offset"] == 26
    
    chunk_restored = Chunk.from_dict(chunk_dict)
    assert chunk_restored.id == chunk.id
    assert chunk_restored.text == chunk.text
    assert chunk_restored.metadata == chunk.metadata
    assert chunk_restored.start_offset == chunk.start_offset
    assert chunk_restored.end_offset == chunk.end_offset


def test_chunk_optional_offsets():
    """Test Chunk with optional offsets."""
    chunk = Chunk(id="chunk_2", text="No offsets.")
    assert chunk.start_offset is None
    assert chunk.end_offset is None
    
    chunk_dict = chunk.to_dict()
    chunk_restored = Chunk.from_dict(chunk_dict)
    assert chunk_restored.start_offset is None
    assert chunk_restored.end_offset is None
