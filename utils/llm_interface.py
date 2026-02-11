from typing import Dict, List, Optional, Tuple
import random
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .models import Component, Entity


class LLMEvaluator:
    """Interface for evaluating component values using LLM."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", mock_api: bool = False):
        """
        Initialize LLM evaluator.
        
        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            model: Model name to use
            mock_api: If True, returns random values instead of calling LLM
        """
        self.mock_api = mock_api
        if not mock_api:
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0
            )
        self._component_cache: Dict[str, Tuple[float, float]] = {}
    
    def _get_cache_key(self, component: Component, entity_ids: List[str], query: str) -> str:
        """Generate cache key for component value."""
        sorted_ids = tuple(sorted(entity_ids))
        return f"{component.name}:{sorted_ids}:{query}"

    def get_component_value_if_cached(
        self, component: Component, entity_ids: List[str], query: str
    ) -> Optional[Tuple[float, float]]:
        """Return (lower, upper) if this component value is cached, else None. Does not call LLM."""
        key = self._get_cache_key(component, entity_ids, query)
        return self._component_cache.get(key)
    
    def _parse_llm_response(self, response_content: str) -> Tuple[float, float]:
        """
        Parse LLM response to extract lower_bound and upper_bound.
        
        Args:
            response_content: The raw response content from LLM
            
        Returns:
            (lower_bound, upper_bound) tuple, both rounded to 1 decimal place
        """
        import re
        
        try:
            # Parse response to extract two floats
            content = response_content.strip()
            # Try splitting by comma
            parts = content.split(',')
            if len(parts) >= 2:
                lb = float(parts[0].strip())
                ub = float(parts[1].strip())
            else:
                # Fallback: extract all numbers
                numbers = re.findall(r'-?\d+\.?\d*', content)
                if len(numbers) >= 2:
                    lb = float(numbers[0])
                    ub = float(numbers[1])
                else:
                    lb = float(numbers[0]) if numbers else 0.0
                    ub = lb  # If only one number, use it for both
        except (ValueError, IndexError):
            # Fallback if parsing fails
            numbers = re.findall(r'-?\d+\.?\d*', response_content)
            if len(numbers) >= 2:
                lb = float(numbers[0])
                ub = float(numbers[1])
            else:
                lb = float(numbers[0]) if numbers else 0.0
                ub = lb
        
        # Clamp to [0, 1] range and ensure lb <= ub
        lb = max(0.0, min(1.0, lb))
        ub = max(0.0, min(1.0, ub))
        if lb > ub:
            lb, ub = ub, lb  # Swap if needed
        
        # Round to 1 decimal place
        lb = round(lb, 1)
        ub = round(ub, 1)
        
        return (lb, ub)
    
    def evaluate_component(
        self,
        component: Component,
        entities: Dict[str, Entity],
        entity_ids: List[str],
        query: str,
        use_cache: bool = True
    ) -> Tuple[float, float]:
        """
        Evaluate a component value using LLM, returning an interval [lower_bound, upper_bound].
        
        Args:
            component: The component to evaluate
            entities: Dictionary mapping entity_id to Entity
            entity_ids: List of entity IDs (1 for unary, 2 for binary)
            query: User query string
            use_cache: Whether to use cached values
            
        Returns:
            Component value as (lower_bound, upper_bound) tuple
        """
        if component.dimension != len(entity_ids):
            raise ValueError(f"Component dimension {component.dimension} doesn't match {len(entity_ids)} entities")
        
        # Check cache
        cache_key = self._get_cache_key(component, entity_ids, query)
        if use_cache and cache_key in self._component_cache:
            return self._component_cache[cache_key]
        
        # If mock_api is True, return random interval instead of calling LLM
        if self.mock_api:
            # Return exact same value for both bounds
            val = round(random.uniform(0.0, 1.0), 1)
            lb = val
            ub = val  # Same value for both bounds
            value = (lb, ub)
            if use_cache:
                self._component_cache[cache_key] = value
            return value
        
        # Build prompt
        entity_info = []
        for i, eid in enumerate(entity_ids):
            entity = entities[eid]
            entity_info.append(f"Entity {i+1} (ID: {entity.id}, Name: {entity.name}, Data: {entity.data})")
        
        # Build comprehensive system prompt
        if component.dimension == 1:
            dimension_explanation = f"""This is a UNARY component, meaning you need to evaluate the {component.name} component value for a SINGLE entity with respect to the user query. You will be given one entity and a user query. Your task is to assess how well this entity satisfies the {component.name} criterion in the context of the query."""
        else:
            dimension_explanation = f"""This is a BINARY component, meaning you need to evaluate the {component.name} component value between TWO entities. You will be given two entities. Your task is to assess the {component.name} relationship or comparison between these two entities."""
        
        system_prompt = f"""You are part of a top-k retrieval system for multimodal data, formulated as a package query problem. In this system, we need to find the best set of k entities that maximize an objective function composed of multiple components.

Your specific role is to evaluate ONE component value, which we call "{component.name}".

Component Details:
- Component Name: {component.name}
- Component Description: {component.description}
- Component Dimension: {component.dimension} ({'unary' if component.dimension == 1 else 'binary'})

{dimension_explanation}

CRITICAL REQUIREMENTS:
1. You must return a RANGE (lower bound and upper bound) as two floating-point numbers
2. Both values MUST be in the range [0, 1], and lower_bound <= upper_bound
3. The format should be: lower_bound, upper_bound (two numbers separated by a comma)
4. IMPORTANT: The lower bound and upper bound MUST be EXACTLY THE SAME value (e.g., "0.5, 0.5" or "0.0, 0.0" or "1.0, 1.0")
5. Do NOT include any explanation, reasoning, or additional text
6. Return ONLY the two numeric values separated by a comma

Your response should be two floats between 0 and 1 separated by a comma where both values are EXACTLY THE SAME (e.g., "0.5, 0.5"), nothing else."""

        human_prompt = f"""User Query: {query}

Entity Information:
{chr(10).join(entity_info)}

Evaluate the {component.name} component value. Return only two float numbers (lower_bound, upper_bound) separated by a comma, both in range [0, 1]. IMPORTANT: Both values must be EXACTLY THE SAME (e.g., "0.5, 0.5"):"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        # Query LLM
        response = self.llm.invoke(prompt.format_messages())
        value = self._parse_llm_response(response.content)
        
        # Cache result
        if use_cache:
            self._component_cache[cache_key] = value
        
        return value
