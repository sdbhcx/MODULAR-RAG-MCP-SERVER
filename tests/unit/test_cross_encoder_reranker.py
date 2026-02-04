"""Unit tests for Cross-Encoder Reranker."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker


class FakeScorer:
    def __init__(self, scores: Dict[str, float]):
        self.scores = scores
        self.calls = []

    def __call__(self, candidate: Dict[str, Any], **kwargs: Any) -> float:
        cid = str(candidate.get('id'))
        self.calls.append(cid)
        return float(self.scores.get(cid, 0.0))


def test_rerank_multiple_candidates_preserves_structure(test_settings):
    scores = {'a': 1.0, 'b': 3.0}
    scorer = FakeScorer(scores)
    reranker = CrossEncoderReranker(settings=test_settings, scorer=scorer)

    candidates = [
        {'id': 'a', 'text': 'A text', 'meta': 1},
        {'id': 'b', 'text': 'B text', 'meta': 2},
    ]

    result = reranker.rerank(query='q', candidates=candidates)

    assert result[0]['id'] == 'b'
    assert result[1]['id'] == 'a'
    assert result[0]['meta'] == 2


def test_rerank_handles_missing_scorer_returns_original_order(test_settings):
    reranker = CrossEncoderReranker(settings=test_settings, scorer=None)
    candidates = [
        {'id': 'x', 'text': 'X'},
        {'id': 'y', 'text': 'Y'},
    ]

    result = reranker.rerank(query='q', candidates=candidates)
    assert result == candidates


def test_rerank_handles_scorer_exceptions(test_settings):
    def bad_scorer(candidate, **kwargs):
        raise RuntimeError('boom')

    reranker = CrossEncoderReranker(settings=test_settings, scorer=bad_scorer)
    candidates = [
        {'id': '1', 'text': 'one'},
        {'id': '2', 'text': 'two'},
    ]

    # Should fallback to original order on scorer internal failure
    result = reranker.rerank(query='q', candidates=candidates)
    assert result == candidates


def test_sort_handles_unscored_candidates(test_settings):
    scores = {'1': 2.0}
    scorer = FakeScorer(scores)
    reranker = CrossEncoderReranker(settings=test_settings, scorer=scorer)

    candidates = [
        {'id': '1', 'text': 'T1'},
        {'id': '2', 'text': 'T2'},
        {'id': '3', 'text': 'T3'},
    ]

    result = reranker.rerank(query='q', candidates=candidates)
    assert result[0]['id'] == '1'
    assert result[1]['id'] == '2'
    assert result[2]['id'] == '3'
