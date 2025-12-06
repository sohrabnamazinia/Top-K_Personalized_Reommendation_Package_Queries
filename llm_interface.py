from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from models import Component, Entity


class LLMEvaluator:
    """Interface for evaluating component values using LLM."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize LLM evaluator.
        
        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            model: Model name to use
        """
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0
        )
        self._component_cache: Dict[str, float] = {}
    
    def _get_cache_key(self, component: Component, entity_ids: List[str], query: str) -> str:
        """Generate cache key for component value."""
        sorted_ids = tuple(sorted(entity_ids))
        return f"{component.name}:{sorted_ids}:{query}"
    
    def evaluate_component(
        self,
        component: Component,
        entities: Dict[str, Entity],
        entity_ids: List[str],
        query: str,
        use_cache: bool = True
    ) -> float:
        """
        Evaluate a component value using LLM.
        
        Args:
            component: The component to evaluate
            entities: Dictionary mapping entity_id to Entity
            entity_ids: List of entity IDs (1 for unary, 2 for binary)
            query: User query string
            use_cache: Whether to use cached values
            
        Returns:
            Component value as float
        """
        if component.dimension != len(entity_ids):
            raise ValueError(f"Component dimension {component.dimension} doesn't match {len(entity_ids)} entities")
        
        # Check cache
        cache_key = self._get_cache_key(component, entity_ids, query)
        if use_cache and cache_key in self._component_cache:
            return self._component_cache[cache_key]
        
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
1. You must return ONLY a single floating-point number
2. The value MUST be in the range [0, 1] (0 = lowest, 1 = highest)
3. Do NOT include any explanation, reasoning, or additional text
4. Return ONLY the numeric value

Your response should be a single float between 0 and 1, nothing else."""

        human_prompt = f"""User Query: {query}

Entity Information:
{chr(10).join(entity_info)}

Evaluate the {component.name} component value. Return only a float number in range [0, 1]:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        # Query LLM
        response = self.llm.invoke(prompt.format_messages())
        try:
            value = float(response.content.strip())
            # Clamp to [0, 1] range
            value = max(0.0, min(1.0, value))
        except ValueError:
            # Fallback if LLM doesn't return pure number
            import re
            numbers = re.findall(r'-?\d+\.?\d*', response.content)
            value = float(numbers[0]) if numbers else 0.0
            # Clamp to [0, 1] range
            value = max(0.0, min(1.0, value))
        
        # Cache result
        if use_cache:
            self._component_cache[cache_key] = value
        
        return value

