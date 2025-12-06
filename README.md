# Top-K Retrieval on Multimodal Data Using Large Language Models

Implementation of an approximate algorithm (AQA: Approximate Algorithm with Chebyshev-Guided Swaps) for top-k retrieval on multimodal data, formulated as a package query problem.

## Overview

This package implements an efficient approximate algorithm to find the top-k entities that maximize a scoring function based on multiple components (e.g., relevance, diversity). The algorithm uses Large Language Models (LLMs) for component value evaluation and employs Chebyshev's inequality to guide efficient candidate selection.

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Structure

- `models.py`: Data models (Entity, Package, Component)
- `llm_interface.py`: LLM interface for component evaluation
- `scoring.py`: Scoring function and contribution calculation
- `algorithm.py`: AQA algorithm implementation
- `load_data.py`: Utility to load entities from CSV
- `test_scoring.py`: Test file 1 - data model and scoring function
- `test_algorithm.py`: Test file 2 - full algorithm with logging

## Usage

### Basic Example

```python
from models import Entity, Component
from llm_interface import LLMEvaluator
from algorithm import AQAAlgorithm

# Define components
components = [
    Component(name='relevance', description='Relevance with query', dimension=1),
    Component(name='diversity', description='Diversity between entities', dimension=2)
]

# Initialize
llm_evaluator = LLMEvaluator()
algorithm = AQAAlgorithm(components, llm_evaluator, budget_rate=5, epsilon=0.01)

# Run algorithm
final_package, metadata = algorithm.run(entities, k=3, query="your query")
```

## Data Format

Entities can be loaded from CSV with columns: `entity_id`, `entity_name`, `data`

See `sample_data.csv` for an example.
