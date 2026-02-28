"""Ingestion Pipeline - Offline data ingestion.

This package contains the data ingestion pipeline:
- Document loading
- Text splitting
- Transform (enrichment)
- Embedding
- Storage
"""

from src.ingestion.pipeline import IngestionPipeline

__all__ = ["IngestionPipeline"]

