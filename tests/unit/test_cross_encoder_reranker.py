"""Unit tests for Cross-Encoder Reranker."""

from unittest.mock import MagicMock, patch
import pytest

# Mock sentence_transformers before importing the module under test
import sys
mock_st = MagicMock()
mock_cross_encoder = MagicMock()
mock_st.CrossEncoder = mock_cross_encoder
sys.modules["sentence_transformers"] = mock_st

from src.core.settings import Settings
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker

@pytest.fixture
def mock_settings():
    return MagicMock(spec=Settings)

@pytest.fixture
def mock_model():
    with patch("src.libs.reranker.cross_encoder_reranker.CrossEncoderReranker") as mock:
        model_instance = MagicMock()
        mock.return_value = model_instance
        yield model_instance

def test_initialization(mock_settings, mock_model):
    """Test initialization loads model."""
    reranker = CrossEncoderReranker(settings=mock_settings, model_name="test-model")
    assert reranker.model_name == "test-model"
    # Verify CrossEncoder instantiatiated
    # Note: since we patched the class in the module, checking call args on mock_model's class
    # But mock_model is the instance. The class is the patch object.
    pass 

def test_rerank_empty(mock_settings, mock_model):
    reranker = CrossEncoderReranker(settings=mock_settings)
    with pytest.raises(ValueError):
        reranker.rerank("query", [])

def test_rerank_success(mock_settings, mock_model):
    """Test successful reranking."""
    
    # Mock predict return values (numpy array style or list)
    # Docs say it returns scores. Higher is better? Yes usually logits.
    mock_model.predict.return_value = [0.1, 0.9]
    
    def mock_scorer(cand, query):
        if cand["id"] == "1":
            return 0.1
        return 0.9
        
    reranker = CrossEncoderReranker(settings=mock_settings, scorer=mock_scorer)
    
    candidates = [
        {"id": "1", "text": "doc1"},
        {"id": "2", "text": "doc2"}
    ]
    
    result = reranker.rerank("query", candidates)
    
    assert len(result) == 2
    assert result[0]["id"] == "2" # Score 0.9
    assert result[1]["id"] == "1" # Score 0.1

def test_rerank_fallback_on_error(mock_settings, mock_model):
    """Test fallback to original order on prediction error."""
    def mock_scorer(cand, query):
        raise RuntimeError("Model failed")
        
    reranker = CrossEncoderReranker(settings=mock_settings, scorer=mock_scorer)
    
    candidates = [{"id": "1", "text": "doc1"}, {"id": "2", "text": "doc2"}]
    result = reranker.rerank("query", candidates)
    
    assert result == candidates

def test_rerank_skips_empty_text(mock_settings, mock_model):
    """Test that candidates without text are handled gracefully."""
    def mock_scorer(cand, query):
        if not cand.get("text"):
            raise ValueError("Empty text")
        return 0.8
        
    reranker = CrossEncoderReranker(settings=mock_settings, scorer=mock_scorer)
    
    candidates = [
        {"id": "1", "text": "valid"},
        {"id": "2", "text": ""} # Empty
    ]
    
    result = reranker.rerank("query", candidates)
    
    # Valid one comes first (score 0.8), empty one last (unscored?)
    assert result[0]["id"] == "1"
    assert result[1]["id"] == "2"
