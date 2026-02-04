from typing import Any, Dict, List

import pytest

from src.libs.vector_store.chroma_store import ChromaStore


def test_chroma_upsert_and_query_basic():
    store = ChromaStore()

    records = [
        {'id': 'r1', 'vector': [1.0, 0.0], 'metadata': {'source': 'a'}},
        {'id': 'r2', 'vector': [0.0, 1.0], 'metadata': {'source': 'b'}},
    ]

    store.upsert(records)

    # Query near r1
    results = store.query([0.9, 0.1], top_k=2)
    assert results
    assert results[0]['id'] == 'r1'


def test_chroma_query_filters():
    store = ChromaStore()

    records = [
        {'id': 'r1', 'vector': [1.0], 'metadata': {'source': 'a'}},
        {'id': 'r2', 'vector': [0.5], 'metadata': {'source': 'b'}},
    ]

    store.upsert(records)

    results = store.query([1.0], top_k=10, filters={'source': 'a'})
    assert len(results) == 1
    assert results[0]['id'] == 'r1'


def test_chroma_delete_and_clear():
    store = ChromaStore()
    store.upsert([{'id': 'r1', 'vector': [1.0]}])
    assert store.query([1.0])

    store.delete(['r1'])
    assert store.query([1.0]) == []

    store.upsert([{'id': 'r2', 'vector': [1.0]}])
    store.clear()
    assert store.query([1.0]) == []
