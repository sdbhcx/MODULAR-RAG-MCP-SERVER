"""
Embedding Module.

This package contains embedding service abstractions and implementations:
- Base embedding class
- Embedding factory
- Provider implementations (OpenAI, Local/BGE)
"""

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
]
