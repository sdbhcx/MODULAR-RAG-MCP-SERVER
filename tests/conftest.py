"""Pytest configuration and shared fixtures.

This module contains pytest configuration and fixtures that are shared
across all test modules.
"""

import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory path.
    
    Returns:
        Path to the project root directory.
    """
    return PROJECT_ROOT


@pytest.fixture
def sample_documents_dir(project_root: Path) -> Path:
    """Return the sample documents directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the sample documents directory.
    """
    return project_root / "tests" / "fixtures" / "sample_documents"


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the config directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the config directory.
    """
    return project_root / "config"


@pytest.fixture
def test_settings() -> MagicMock:
    """Provide a lightweight mock Settings object for unit tests."""
    settings = MagicMock()
    settings.llm = MagicMock()
    settings.llm.provider = "openai"
    settings.rerank = MagicMock()
    settings.rerank.enabled = True
    settings.rerank.provider = "llm"
    return settings
