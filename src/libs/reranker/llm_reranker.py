"""LLM-based Reranker implementation.

This module provides an implementation of the Reranker interface using LLM models
to score and rank candidate passages based on relevance to the user's query.
The reranker reads a prompt template from config/prompts/rerank.txt and uses
an LLM to evaluate passage relevance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

# Default prompt template path
DEFAULT_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "config" / "prompts" / "rerank.txt"


class LLMReranker(BaseReranker):
    """LLM-based reranker that uses an LLM to score candidate passages.
    
    This implementation uses a Language Model to evaluate the relevance of candidate
    passages to a given query. It reads a prompt template from config/prompts/rerank.txt
    and constructs a structured prompt that asks the LLM to rank candidates.
    
    Design Principles Applied:
    - Pluggable: Implements BaseReranker interface, swappable with other rerankers.
    - Config-Driven: LLM provider and settings come from settings.yaml.
    - Observable: Accepts optional TraceContext for observability integration.
    - Fail-Safe: Returns candidates in original order on rerank failure.
    - Testable: Supports dependency injection of LLM instance for testing.
    
    Attributes:
        llm: The LLM instance used for scoring.
        prompt_template: The prompt template read from config file.
        settings: Application settings.
        max_retries: Maximum retries on LLM failure.
        timeout: Timeout for LLM calls in seconds.
    
    Example:
        >>> from src.core.settings import Settings
        >>> from src.libs.llm.llm_factory import LLMFactory
        >>> settings = Settings.load('config/settings.yaml')
        >>> reranker = LLMReranker(settings=settings)
        >>> candidates = [
        ...     {'id': '1', 'text': 'First passage about Python'},
        ...     {'id': '2', 'text': 'Second passage about programming'}
        ... ]
        >>> reranked = reranker.rerank(query="What is Python?", candidates=candidates)
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_template: Optional[str] = None,
        max_retries: int = 1,
        timeout: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize LLMReranker with configuration.
        
        Args:
            settings: Application settings containing LLM configuration.
            llm: Optional LLM instance. If None, will be created from settings.
            prompt_template: Optional custom prompt template. If None, loaded from config file.
            max_retries: Maximum retries on LLM failure (default: 1, no retries).
            timeout: Timeout for LLM calls in seconds (optional).
            **kwargs: Additional parameters (reserved for future use).
        
        Raises:
            ValueError: If required configuration is missing or LLM initialization fails.
            FileNotFoundError: If prompt template file is not found.
        """
        self.settings = settings
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize LLM
        if llm is not None:
            # Use injected LLM (for testing)
            self.llm = llm
            logger.debug("LLMReranker initialized with injected LLM instance")
        else:
            # Create LLM from factory
            try:
                from src.libs.llm.llm_factory import LLMFactory
                self.llm = LLMFactory.create(settings)
                logger.debug(f"LLMReranker initialized with LLM provider: {type(self.llm).__name__}")
            except Exception as e:
                raise ValueError(
                    f"Failed to initialize LLM for reranker: {e}. "
                    "Please ensure LLM configuration is valid in settings.yaml"
                ) from e
        
        # Load prompt template
        if prompt_template is not None:
            self.prompt_template = prompt_template
            logger.debug("LLMReranker using custom prompt template")
        else:
            # Load from config file
            try:
                self.prompt_template = self._load_prompt_template()
                logger.debug(f"LLMReranker loaded prompt template from {DEFAULT_PROMPT_PATH}")
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Rerank prompt template not found at {DEFAULT_PROMPT_PATH}. "
                    "Please ensure config/prompts/rerank.txt exists."
                ) from e
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate chunks using an LLM.
        
        Args:
            query: The user query string.
            candidates: List of candidate records to rerank. Each record should have:
                - 'id': Unique identifier (str)
                - 'text': The text content to evaluate (str)
                - Other fields are preserved in output
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Provider-specific parameters (e.g., timeout override).
        
        Returns:
            List of candidates sorted by LLM-assigned relevance score (descending).
            Each candidate retains its original structure but is reordered.
            If reranking fails, returns candidates in original order.
        
        Raises:
            ValueError: If query or candidates are invalid.
        """
        # Validate inputs
        self.validate_query(query)
        self.validate_candidates(candidates)
        
        # If only 1 candidate, no need to rerank
        if len(candidates) <= 1:
            return candidates
        
        try:
            # Construct rerank prompt
            rerank_prompt = self._construct_rerank_prompt(query, candidates)
            
            # Call LLM to get scores
            logger.debug(f"Calling LLM to rerank {len(candidates)} candidates")
            scores = self._get_llm_scores(rerank_prompt, len(candidates), **kwargs)
            
            # Sort candidates by scores (descending)
            reranked = self._sort_by_scores(candidates, scores)
            
            logger.info(f"Successfully reranked {len(candidates)} candidates")
            return reranked
            
        except Exception as e:
            # Fallback: return candidates in original order
            logger.warning(
                f"LLM reranking failed: {e}. Returning candidates in original order. "
                "This is a fallback behavior - check LLM configuration."
            )
            return candidates
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from config file.
        
        Returns:
            The prompt template string.
        
        Raises:
            FileNotFoundError: If the template file is not found.
        """
        prompt_path = DEFAULT_PROMPT_PATH
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            return template
        except IOError as e:
            raise FileNotFoundError(f"Failed to read prompt template: {e}") from e
    
    def _construct_rerank_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """Construct the rerank prompt for the LLM.
        
        Args:
            query: The user query.
            candidates: List of candidate records with 'id' and 'text' fields.
        
        Returns:
            The constructed prompt string.
        """
        # Build candidate list for prompt
        candidate_text = ""
        for i, candidate in enumerate(candidates, 1):
            candidate_id = candidate.get('id', f'candidate_{i}')
            candidate_content = candidate.get('text', '')
            candidate_text += f"\nCandidate {i} (ID: {candidate_id}):\n{candidate_content}\n---"
        
        # Construct final prompt
        prompt = f"""{self.prompt_template}

Query: {query}

Candidates to rank:{candidate_text}

Please score each candidate and return the results in valid JSON format as described above.
Ensure the JSON is parseable and contains exactly {len(candidates)} objects."""
        
        return prompt
    
    def _get_llm_scores(
        self,
        prompt: str,
        num_candidates: int,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Get LLM scores for candidates.
        
        Args:
            prompt: The prompt to send to the LLM.
            num_candidates: Number of candidates being scored.
            **kwargs: Additional parameters (e.g., timeout).
        
        Returns:
            Dict mapping candidate IDs to scores (0-3 scale).
        
        Raises:
            ValueError: If LLM response is not valid JSON or missing required fields.
            RuntimeError: If LLM call fails after retries.
        """
        timeout = kwargs.get('timeout', self.timeout)
        last_error = None
        
        # Retry loop
        for attempt in range(self.max_retries + 1):
            try:
                # Call LLM
                response = self.llm.generate(
                    prompt=prompt,
                    max_tokens=2000,
                    temperature=0.1,  # Low temperature for consistent scoring
                    timeout=timeout
                )
                
                # Parse response
                scores = self._parse_llm_response(response, num_candidates)
                return scores
                
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"LLM response parsing failed (attempt {attempt + 1}), retrying: {e}")
                    continue
                else:
                    raise ValueError(f"Failed to parse LLM response after {attempt + 1} attempts: {e}") from e
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying: {e}")
                    continue
                else:
                    raise RuntimeError(f"LLM call failed after {attempt + 1} attempts: {e}") from e
        
        # Should not reach here, but handle gracefully
        raise RuntimeError(f"LLM reranking failed: {last_error}")
    
    def _parse_llm_response(self, response: str, num_candidates: int) -> Dict[str, float]:
        """Parse LLM response to extract scores.
        
        The LLM should return JSON with objects containing 'passage_id' and 'score'.
        
        Args:
            response: The raw LLM response string.
            num_candidates: Expected number of scored candidates.
        
        Returns:
            Dict mapping candidate IDs to scores.
        
        Raises:
            ValueError: If response format is invalid.
            json.JSONDecodeError: If response is not valid JSON.
        """
        # Extract JSON from response (LLM might include extra text)
        try:
            # Try to extract JSON array
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                # Try single object
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    raise ValueError("No valid JSON found in LLM response")
                
                # Single object, wrap in array
                json_str = f"[{response[json_start:json_end]}]"
            else:
                json_str = response[json_start:json_end]
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Ensure it's a list
            if not isinstance(parsed, list):
                parsed = [parsed]
            
            # Extract scores
            scores = {}
            for item in parsed:
                if not isinstance(item, dict):
                    raise ValueError(f"Expected dict in response, got {type(item).__name__}")
                
                passage_id = item.get('passage_id')
                score = item.get('score')
                
                if passage_id is None:
                    raise ValueError("Missing 'passage_id' in LLM response")
                if score is None:
                    raise ValueError("Missing 'score' in LLM response")
                
                try:
                    score = float(score)
                    if not 0 <= score <= 3:
                        logger.warning(f"Score {score} for {passage_id} is outside 0-3 range, clamping")
                        score = max(0.0, min(3.0, score))
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid score value '{score}' for {passage_id}: {e}") from e
                
                scores[str(passage_id)] = score
            
            # Verify we got all expected scores
            if len(scores) != num_candidates:
                logger.warning(
                    f"Expected scores for {num_candidates} candidates, got {len(scores)}. "
                    "Some candidates may have been skipped by the LLM."
                )
            
            return scores
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}") from e
    
    def _sort_by_scores(
        self,
        candidates: List[Dict[str, Any]],
        scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Sort candidates by LLM-assigned scores.
        
        Args:
            candidates: Original list of candidates.
            scores: Dict mapping candidate IDs to scores.
        
        Returns:
            Candidates sorted by score in descending order.
            Candidates without scores are placed at the end in original order.
        """
        scored = []
        unscored = []
        
        for candidate in candidates:
            candidate_id = str(candidate.get('id', ''))
            if candidate_id in scores:
                scored.append((candidate, scores[candidate_id]))
            else:
                unscored.append(candidate)
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Combine: scored first (by score), then unscored
        result = [c for c, _ in scored] + unscored
        
        return result
