from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import base64
import random
import os
import time
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from .models import Component, Entity

if TYPE_CHECKING:
    from preprocessing.MGT import MGT


class LLMEvaluator:
    """Interface for evaluating component values using LLM or materialized ground truth (MGT)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        mock_api: bool = False,
        use_MGT: bool = True,
        entities_csv_path: Optional[str] = None,
        components: Optional[List[Component]] = None,
        output_dir: str = "mgt_Results",
        n: Optional[int] = None,
        images_base_path: Optional[str] = None,
    ):
        """
        Initialize LLM evaluator.

        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            model: Model name to use
            mock_api: If True, returns random values instead of calling LLM
            use_MGT: If True, read component values from MGT CSVs instead of calling LLM.
                     Requires entities_csv_path, components, and n.
            entities_csv_path: Path to entities CSV (required when use_MGT=True)
            components: List of Component (required when use_MGT=True)
            output_dir: Directory for MGT CSVs (default mgt_Results)
            n: Number of entities (required when use_MGT=True). If MGT for n does not exist, it is built by slicing a larger-n MGT if available.
            images_base_path: Optional directory to resolve entity image_id to image files (e.g. for Yelp: path to photos folder).
                             Used only for dimension-1 components when making real LLM calls; if entity has image_id and file exists, image is sent to the LLM.
        """
        self.mock_api = mock_api
        self.use_MGT = use_MGT
        self._mgt: Optional["MGT"] = None
        self.images_base_path = images_base_path

        if use_MGT:
            if not entities_csv_path or not components:
                raise ValueError("use_MGT is True but entities_csv_path and components are required")
            if n is None:
                raise ValueError("use_MGT is True but n (number of entities) is required")
            from preprocessing.MGT import MGT
            self._mgt = MGT(entities_csv_path, components)
            self._mgt.ensure_mgt_for_n(output_dir, n)
            self._mgt.load_from_existing(output_dir, n=n)
        elif not mock_api:
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0
            )
        self._component_cache: Dict[str, Tuple[float, float, float]] = {}
    
    def _get_cache_key(self, component: Component, entity_ids: List[str], query: str) -> str:
        """Generate cache key for component value."""
        sorted_ids = tuple(sorted(entity_ids))
        return f"{component.name}:{sorted_ids}:{query}"

    def get_component_value_if_cached(
        self, component: Component, entity_ids: List[str], query: str
    ) -> Optional[Tuple[float, float, float]]:
        """Return (lower, upper, time) if this component value is cached, else None. Does not call LLM."""
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
        print(response_content)  # Debug print to see raw response
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
        use_cache: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Evaluate a component value: from MGT if use_MGT else from LLM.
        Returns (lower_bound, upper_bound, time_taken) tuple.
        """
        if component.dimension != len(entity_ids):
            raise ValueError(f"Component dimension {component.dimension} doesn't match {len(entity_ids)} entities")

        cache_key = self._get_cache_key(component, entity_ids, query)
        if use_cache and cache_key in self._component_cache:
            return self._component_cache[cache_key]
        if self.use_MGT and self._mgt is not None:
            value = self._evaluate_component_mgt(component, entity_ids, use_cache, cache_key)
        else:
            value = self._evaluate_component_llm(component, entities, entity_ids, query, use_cache, cache_key)
        return value

    def _evaluate_component_mgt(
        self,
        component: Component,
        entity_ids: List[str],
        use_cache: bool,
        cache_key: str,
    ) -> Tuple[float, float, float]:
        """Read (lb, ub, time) from MGT CSV and optionally cache."""
        lb, ub, time_val = self._mgt.fetch(component.name, entity_ids)
        value = (float(lb), float(ub), float(time_val))
        if use_cache:
            self._component_cache[cache_key] = value
        return value

    def _evaluate_component_llm(
        self,
        component: Component,
        entities: Dict[str, Entity],
        entity_ids: List[str],
        query: str,
        use_cache: bool,
        cache_key: str,
    ) -> Tuple[float, float, float]:
        """Call LLM to evaluate component; returns (lower_bound, upper_bound, time_taken)."""
        t0 = time.perf_counter()
        if self.mock_api:
            val = round(random.uniform(0.0, 1.0), 1)
            lb = ub = val
            elapsed = time.perf_counter() - t0
            value = (lb, ub, round(elapsed, 4))
            if use_cache:
                self._component_cache[cache_key] = value
            return value

        # Escape curly braces so LangChain template parser treats them as literal (e.g. Wikipedia {{...}} in entity data)
        def _escape_braces(s: str) -> str:
            return (s or "").replace("{", "{{").replace("}", "}}")

        # Build prompt
        entity_info = []
        for i, eid in enumerate(entity_ids):
            entity = entities[eid]
            entity_info.append(f"Entity {i+1} (ID: {_escape_braces(entity.id)}, Name: {_escape_braces(entity.name)}, Data: {_escape_braces(entity.data)})")

        query_esc = _escape_braces(query)
        desc_esc = _escape_braces(component.description)

        # Build comprehensive system prompt
        if component.dimension == 1:
            dimension_explanation = f"""This is a UNARY component, meaning you need to evaluate the {component.name} component value for a SINGLE entity with respect to the user query. You will be given one entity and a user query. Your task is to assess how well this entity satisfies the {component.name} criterion in the context of the query."""
        else:
            dimension_explanation = f"""This is a BINARY component, meaning you need to evaluate the {component.name} component value between TWO entities. You will be given two entities. Your task is to assess the {component.name} relationship or comparison between these two entities."""

        system_prompt = f"""You are part of a top-k retrieval system for multimodal data, formulated as a package query problem. In this system, we need to find the best set of k entities that maximize an objective function composed of multiple components.

Your specific role is to evaluate ONE component value, which we call "{component.name}".

Component Details:
- Component Name: {component.name}
- Component Description: {desc_esc}
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

        human_prompt = f"""User Query: {query_esc}

Entity Information:
{chr(10).join(entity_info)}

Evaluate the {component.name} component value. Return only two float numbers (lower_bound, upper_bound) separated by a comma, both in range [0, 1]. IMPORTANT: Both values must be EXACTLY THE SAME (e.g., "0.5, 0.5"):"""

        # Dimension 1 + real LLM + entity has image_id + images_base_path set: include image in the request
        use_image = (
            component.dimension == 1
            and self.images_base_path
            and len(entity_ids) == 1
            and getattr(entities[entity_ids[0]], "image_id", None)
        )
        image_data_url = None
        if use_image:
            entity = entities[entity_ids[0]]
            image_id = getattr(entity, "image_id", None)
            if image_id:
                base_path = Path(self.images_base_path)
                for ext in (".jpg", ".jpeg", ".png", ""):
                    img_path = base_path / f"{image_id}{ext}" if ext else base_path / image_id
                    if img_path.is_file():
                        try:
                            b64 = base64.standard_b64encode(img_path.read_bytes()).decode("ascii")
                            mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                            image_data_url = f"data:{mime};base64,{b64}"
                        except Exception:
                            pass
                        break

        if image_data_url:
            content = [
                {"type": "text", "text": human_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=content)]
            response = self.llm.invoke(messages)
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            response = self.llm.invoke(prompt.format_messages())
        
        lb, ub = self._parse_llm_response(response.content)
        elapsed = time.perf_counter() - t0
        value = (lb, ub, round(elapsed, 4))
        if use_cache:
            self._component_cache[cache_key] = value
        return value
