"""Unit tests for LLM Reranker implementation.

Tests cover:
- Reranking with mock LLM
- Prompt template loading and construction
- Error handling and fallback behavior
- Score parsing from LLM response
- Edge cases (empty candidates, single candidate, etc.)
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.llm_reranker import LLMReranker


class FakeLLM:
    """Fake LLM for testing."""
    
    def __init__(self, response: str = "", should_fail: bool = False):
        self.response = response
        self.should_fail = should_fail
        self.call_count = 0
        self.last_prompt = None
    
    def chat(self, messages: List[Any], **kwargs: Any) -> Any:
        self.call_count += 1
        # Extract user prompt
        for msg in reversed(messages):
            if msg.role == 'user':
                self.last_prompt = msg.content
                break
        
        if self.should_fail:
            raise RuntimeError("LLM generation failed")
        
        return type('ChatResponse', (), {'content': self.response, 'model': 'test'})


class TestLLMRerankerInitialization:
    """Tests for LLMReranker initialization."""
    
    def test_init_with_injected_llm(self, test_settings):
        """Test initialization with injected LLM instance."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        assert reranker.llm is fake_llm
        assert reranker.prompt_template is not None
        assert len(reranker.prompt_template) > 0
    
    def test_init_with_custom_prompt(self, test_settings):
        """Test initialization with custom prompt template."""
        fake_llm = FakeLLM()
        custom_prompt = "Custom reranking prompt"
        
        reranker = LLMReranker(
            settings=test_settings,
            llm=fake_llm,
            prompt_template=custom_prompt
        )
        
        assert reranker.prompt_template == custom_prompt
    
    def test_init_loads_default_prompt(self, test_settings):
        """Test that initialization loads default prompt from config."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        # Should contain content from rerank.txt
        assert "relevance" in reranker.prompt_template.lower() or \
               "score" in reranker.prompt_template.lower()
    
    def test_init_with_timeout(self, test_settings):
        """Test initialization with timeout parameter."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm, timeout=30)
        
        assert reranker.timeout == 30


class TestLLMRerankerRanking:
    """Tests for the core reranking functionality."""
    
    def test_rerank_single_candidate(self, test_settings):
        """Test that single candidate returns unchanged."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        candidates = [{'id': '1', 'text': 'Passage about Python'}]
        result = reranker.rerank(query="What is Python?", candidates=candidates)
        
        assert result == candidates
        # LLM should not be called for single candidate
        assert fake_llm.call_count == 0
    
    def test_rerank_multiple_candidates(self, test_settings):
        """Test reranking of multiple candidates."""
        # Mock LLM response with scores
        llm_response = json.dumps([
            {'passage_id': '1', 'score': 3, 'reasoning': 'Directly answers the query'},
            {'passage_id': '2', 'score': 1, 'reasoning': 'Marginally related'}
        ])
        
        fake_llm = FakeLLM(response=llm_response)
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        candidates = [
            {'id': '2', 'text': 'Marginally related passage'},
            {'id': '1', 'text': 'Passage directly about Python'}
        ]
        
        result = reranker.rerank(query="What is Python?", candidates=candidates)
        
        # Should be reordered: higher score first
        assert result[0]['id'] == '1'
        assert result[1]['id'] == '2'
    
    def test_rerank_preserves_candidate_structure(self, test_settings):
        """Test that reranking preserves candidate structure."""
        llm_response = json.dumps([
            {'passage_id': 'doc1', 'score': 2},
            {'passage_id': 'doc2', 'score': 3}
        ])
        
        fake_llm = FakeLLM(response=llm_response)
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        candidates = [
            {'id': 'doc1', 'text': 'Text 1', 'metadata': {'source': 'file1'}, 'custom_field': 'value1'},
            {'id': 'doc2', 'text': 'Text 2', 'metadata': {'source': 'file2'}, 'custom_field': 'value2'}
        ]
        
        result = reranker.rerank(query="Test query", candidates=candidates)
        
        # Check that reordered candidate retains all fields
        assert result[0]['id'] == 'doc2'
        assert result[0]['text'] == 'Text 2'
        assert result[0]['metadata'] == {'source': 'file2'}
        assert result[0]['custom_field'] == 'value2'
    
    def test_rerank_handles_llm_failure_gracefully(self, test_settings):
        """Test that reranking falls back to original order on LLM failure."""
        fake_llm = FakeLLM(should_fail=True)
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        candidates = [
            {'id': '1', 'text': 'First passage'},
            {'id': '2', 'text': 'Second passage'}
        ]
        
        result = reranker.rerank(query="Test query", candidates=candidates)
        
        # Should return original order on failure
        assert result == candidates


class TestLLMRerankerPromptConstruction:
    """Tests for prompt template construction."""
    
    def test_construct_rerank_prompt(self, test_settings):
        """Test that prompt construction includes query and candidates."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        query = "What is machine learning?"
        candidates = [
            {'id': 'doc1', 'text': 'Machine learning is a subset of AI'},
            {'id': 'doc2', 'text': 'Deep learning uses neural networks'}
        ]
        
        prompt = reranker._construct_rerank_prompt(query, candidates)
        
        # Prompt should contain query and candidate information
        assert query in prompt
        assert 'Machine learning is' in prompt
        assert 'Deep learning uses' in prompt
        assert 'doc1' in prompt
        assert 'doc2' in prompt
    
    def test_prompt_uses_custom_template(self, test_settings):
        """Test that custom template is used in constructed prompt."""
        custom_template = "CUSTOM_TEMPLATE: Rank these passages"
        fake_llm = FakeLLM()
        reranker = LLMReranker(
            settings=test_settings,
            llm=fake_llm,
            prompt_template=custom_template
        )
        
        query = "Test query"
        candidates = [{'id': '1', 'text': 'Test passage'}]
        
        prompt = reranker._construct_rerank_prompt(query, candidates)
        
        assert custom_template in prompt


class TestLLMRerankerResponseParsing:
    """Tests for LLM response parsing."""
    
    def test_parse_json_array_response(self, test_settings):
        """Test parsing JSON array response."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        response = json.dumps([
            {'passage_id': '1', 'score': 3},
            {'passage_id': '2', 'score': 1}
        ])
        
        scores = reranker._parse_llm_response(response, num_candidates=2)
        
        assert scores['1'] == 3.0
        assert scores['2'] == 1.0
    
    def test_parse_json_with_extra_text(self, test_settings):
        """Test parsing JSON response with extra text around it."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        response = """
        Here are the scores:
        [{"passage_id": "1", "score": 2}, {"passage_id": "2", "score": 3}]
        These are ranked by relevance.
        """
        
        scores = reranker._parse_llm_response(response, num_candidates=2)
        
        assert scores['1'] == 2.0
        assert scores['2'] == 3.0
    
    def test_parse_handles_missing_fields(self, test_settings):
        """Test that parsing raises error on missing required fields."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        # Missing score field
        response = json.dumps([{'passage_id': '1'}])
        
        with pytest.raises(ValueError, match="score"):
            reranker._parse_llm_response(response, num_candidates=1)
    
    def test_parse_clamps_score_range(self, test_settings):
        """Test that scores outside 0-3 range are clamped."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        response = json.dumps([
            {'passage_id': '1', 'score': 5},  # Out of range
            {'passage_id': '2', 'score': -1}  # Out of range
        ])
        
        scores = reranker._parse_llm_response(response, num_candidates=2)
        
        assert scores['1'] == 3.0  # Clamped to max
        assert scores['2'] == 0.0  # Clamped to min
    
    def test_parse_invalid_json_raises_error(self, test_settings):
        """Test that invalid JSON raises appropriate error."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        response = "This is not JSON"
        
        with pytest.raises(ValueError):
            reranker._parse_llm_response(response, num_candidates=1)


class TestLLMRerankerValidation:
    """Tests for input validation."""
    
    def test_rerank_validates_query(self, test_settings):
        """Test that empty query is rejected."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        candidates = [{'id': '1', 'text': 'Passage'}]
        
        with pytest.raises(ValueError, match="empty"):
            reranker.rerank(query="   ", candidates=candidates)
    
    def test_rerank_validates_candidates(self, test_settings):
        """Test that empty candidates list is rejected."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        with pytest.raises(ValueError, match="empty"):
            reranker.rerank(query="Test", candidates=[])
    
    def test_rerank_validates_candidate_structure(self, test_settings):
        """Test that malformed candidates are rejected."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        candidates = ["not a dict"]
        
        with pytest.raises(ValueError):
            reranker.rerank(query="Test", candidates=candidates)


class TestLLMRerankerSorting:
    """Tests for sorting by scores."""
    
    def test_sort_by_scores_descending(self, test_settings):
        """Test that candidates are sorted by score in descending order."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        candidates = [
            {'id': '1', 'text': 'Text 1'},
            {'id': '2', 'text': 'Text 2'},
            {'id': '3', 'text': 'Text 3'}
        ]
        
        scores = {'1': 1.0, '2': 3.0, '3': 2.0}
        
        result = reranker._sort_by_scores(candidates, scores)
        
        assert result[0]['id'] == '2'  # Score 3
        assert result[1]['id'] == '3'  # Score 2
        assert result[2]['id'] == '1'  # Score 1
    
    def test_sort_handles_unscored_candidates(self, test_settings):
        """Test that candidates without scores are placed at end."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        candidates = [
            {'id': '1', 'text': 'Text 1'},
            {'id': '2', 'text': 'Text 2'},
            {'id': '3', 'text': 'Text 3'}
        ]
        
        scores = {'1': 2.0}  # Only score for id 1
        
        result = reranker._sort_by_scores(candidates, scores)
        
        assert result[0]['id'] == '1'
        # Unscored candidates should be at the end in original order
        assert result[1]['id'] == '2'
        assert result[2]['id'] == '3'


class TestLLMRerankerLoadingPrompt:
    """Tests for prompt template loading."""
    
    def test_load_prompt_from_default_path(self, test_settings):
        """Test loading prompt from default config path."""
        fake_llm = FakeLLM()
        reranker = LLMReranker(settings=test_settings, llm=fake_llm)
        
        # Should have loaded prompt successfully
        assert reranker.prompt_template is not None
        assert len(reranker.prompt_template) > 0
    
    def test_load_nonexistent_prompt_raises_error(self, test_settings):
        """Test that missing prompt file raises clear error."""
        fake_llm = FakeLLM()
        
        # Try to initialize with non-existent prompt path
        with patch.object(LLMReranker, '_load_prompt_template') as mock_load:
            mock_load.side_effect = FileNotFoundError("Prompt not found")
            
            with pytest.raises(FileNotFoundError, match="not found"):
                reranker = LLMReranker(settings=test_settings, llm=fake_llm)


# Fixtures
@pytest.fixture
def test_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.llm = MagicMock()
    settings.llm.provider = "openai"
    settings.rerank = MagicMock()
    settings.rerank.enabled = True
    settings.rerank.provider = "llm"
    return settings
