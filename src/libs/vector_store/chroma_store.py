"""Minimal ChromaStore implementation (in-memory) for development & tests.

Provides a simple, file-free in-memory vector store that implements the
`BaseVectorStore` contract. Intended as a default, local fallback for
development and unit tests before integrating a real Chroma backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import os

from src.libs.vector_store.base_vector_store import BaseVectorStore


class ChromaStoreError(RuntimeError):
    """Raised when ChromaStore operations fail."""


class ChromaStore(BaseVectorStore):
    """Simple in-memory vector store implementing BaseVectorStore contract.

    Notes:
        - This implementation is intentionally lightweight and synchronous.
        - Similarity is computed with cosine similarity.
        - Meant for tests and local development; not for production scale.
    """

    def __init__(
        self,
        settings: Any = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        chroma_impl: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.collection_name = collection_name or "default"

        # Runtime: prefer chromadb if available, otherwise fallback to in-memory
        self._using_chroma = False
        self._storage: Dict[str, Dict[str, Any]] = {}

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            impl = chroma_impl or getattr(
                settings.vector_store, "chroma_impl", None
            ) if settings is not None else None
            impl = impl or os.environ.get("CHROMA_IMPL", "duckdb+parquet")

            persist_dir = (
                persist_directory
                or getattr(settings.vector_store, "persist_directory", None)
                or os.environ.get("CHROMA_PERSIST_DIR", None)
            )

            client_settings = ChromaSettings(
                chroma_db_impl=impl,
                persist_directory=persist_dir,
            )

            self._client = chromadb.Client(client_settings)
            # get or create collection
            try:
                self._collection = self._client.get_collection(self.collection_name)
            except Exception:
                self._collection = self._client.create_collection(self.collection_name)

            self._using_chroma = True
        except Exception:
            # chromadb not available or failed to init -> use in-memory fallback
            self._using_chroma = False
            self._storage = {}

    def upsert(self, records: List[Dict[str, Any]], trace: Optional[Any] = None, **kwargs: Any) -> None:
        self.validate_records(records)

        if self._using_chroma:
            try:
                ids = [r['id'] for r in records]
                embeddings = [list(r['vector']) for r in records]
                metadatas = [r.get('metadata', {}) for r in records]

                # chroma collection.add API
                self._collection.add(
                    ids=ids, embeddings=embeddings, metadatas=metadatas
                )
                return
            except Exception as e:
                raise ChromaStoreError(f"Chroma upsert failed: {e}") from e

        for record in records:
            # store a shallow copy to avoid external mutation
            self._storage[record['id']] = {
                'id': record['id'],
                'vector': list(record['vector']),
                'metadata': record.get('metadata', {}),
            }

    def query(self, vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, trace: Optional[Any] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        self.validate_query_vector(vector, top_k)

        if self._using_chroma:
            try:
                # chroma query API expects list of embeddings
                query_result = self._collection.query(
                    query_embeddings=[list(vector)], n_results=top_k, where=filters
                )

                # chroma returns dict with 'ids', 'distances', 'metadatas'
                ids = query_result.get('ids', [[]])[0]
                distances = query_result.get('distances', [[]])[0]
                metadatas = query_result.get('metadatas', [[]])[0]

                results: List[Dict[str, Any]] = []
                for i, _id in enumerate(ids):
                    dist = distances[i] if i < len(distances) else None
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    # convert distance -> score (best-effort)
                    score = None
                    if dist is None:
                        score = 0.0
                    else:
                        try:
                            score = float(1.0 - dist)
                        except Exception:
                            score = float(dist)

                    results.append({'id': _id, 'score': score, 'metadata': metadata, 'vector': None})

                return results
            except Exception as e:
                raise ChromaStoreError(f"Chroma query failed: {e}") from e

        # Fallback: in-memory cosine similarity
        q_norm = math.sqrt(sum(float(x) * float(x) for x in vector))
        if q_norm == 0:
            raise ChromaStoreError("Query vector norm is zero")

        results: List[Dict[str, Any]] = []

        for record_id, rec in self._storage.items():
            # Apply metadata filters if provided
            if filters:
                metadata = rec.get('metadata', {})
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue

            # compute cosine similarity
            vec = rec['vector']
            dot = sum(float(a) * float(b) for a, b in zip(vector, vec))
            v_norm = math.sqrt(sum(float(x) * float(x) for x in vec))
            if v_norm == 0:
                score = 0.0
            else:
                score = dot / (q_norm * v_norm)

            results.append({
                'id': record_id,
                'score': float(score),
                'metadata': rec.get('metadata', {}),
                'vector': rec.get('vector'),
            })

        # Sort by score descending and return top_k
        results.sort(key=lambda r: r['score'], reverse=True)
        return results[:top_k]

    def delete(self, ids: List[str], trace: Optional[Any] = None, **kwargs: Any) -> None:
        if not ids:
            raise ValueError("ids list cannot be empty")
        if self._using_chroma:
            try:
                self._collection.delete(ids=ids)
                return
            except Exception as e:
                raise ChromaStoreError(f"Chroma delete failed: {e}") from e

        for _id in ids:
            self._storage.pop(_id, None)

    def clear(self, collection_name: Optional[str] = None, trace: Optional[Any] = None, **kwargs: Any) -> None:
        if self._using_chroma:
            try:
                # remove and recreate collection to clear
                name = collection_name or self.collection_name
                try:
                    self._client.delete_collection(name)
                except Exception:
                    pass
                self._client.create_collection(name)
                # update reference
                self._collection = self._client.get_collection(name)
                return
            except Exception as e:
                raise ChromaStoreError(f"Chroma clear failed: {e}") from e

        # clear default or specified collection (collections unsupported in this minimal impl)
        self._storage.clear()


# Auto-register with factory for convenience
try:
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    VectorStoreFactory.register_provider('chroma', ChromaStore)
except Exception:
    # Registration is best-effort; tests may register their own providers
    pass
