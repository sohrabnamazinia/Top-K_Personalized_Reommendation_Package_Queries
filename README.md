# Top-K Retrieval on Multimodal Data Using Large Language Models

Implementation of approximate and exact algorithms for top-k retrieval on multimodal data, formulated as a package query problem.

## Overview

This package implements two algorithms for finding the top-k entities that maximize a scoring function based on multiple components (e.g., relevance, diversity):

1. **PCS (Approximate Algorithm)**: Approximate Algorithm with Chebyshev-Guided Swaps - an efficient approximate method
2. **AQS (Exact Algorithm)**: Admissible Question Selection Algorithm - an exact method using interval-based bounds

Both algorithms use Large Language Models (LLMs) for component value evaluation and employ different strategies for efficient candidate selection.

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Project Structure

### Root Folder (Common Files)
- `models.py`: Data models (Entity, Package, Component)
- `llm_interface.py`: LLM interface for component evaluation (returns intervals [lower, upper])
- `preprocessing/load_data.py`: Utility to load entities from CSV
- `sample_data.csv`: Example data file

### PCS Folder (Approximate Algorithm)
- `algorithm.py`: PCS algorithm implementation with Chebyshev-guided swaps
- `scoring.py`: Scoring function and contribution calculation
- `test_scoring_PCS.py`: Test file for scoring function
- `test_algorithm.py`: Test file for full PCS algorithm

### AQS Folder (Exact Algorithm)
- `algorithm.py`: AQS algorithm implementation with admissible question selection
- `scoring.py`: Interval-based scoring function
- `package_manager.py`: PackageManager class for managing candidate packages and bounds
- `test_scoring.py`: Test file for scoring function
- `test_package_manager.py`: Test file for PackageManager
- `test_algorithm.py`: Test file for full AQS algorithm

## Key Features

### Component-Based Scoring
- **Unary Components**: Evaluate a single entity (e.g., relevance with query)
- **Binary Components**: Evaluate a pair of entities (e.g., diversity)
- Component values are returned as intervals [lower_bound, upper_bound] in range [0, 1]

### LLM Integration
- Uses LangChain with OpenAI for component evaluation
- Supports mock API mode for testing (returns random intervals)
- Automatic caching of component values
- Values rounded to 1 decimal place

### Interval-Based Bounds
- Package scores computed as intervals [lower, upper]
- Unknown component values contribute [0, 1] to bounds
- Known component values use their actual intervals
- Bounds updated as questions are answered

## Usage

### PCS Algorithm (Approximate)

```python
from utils.models import Entity, Component
from utils.llm_interface import LLMEvaluator
from PCS.algorithm import PCSAlgorithm

# Define components
components = [
    Component(name='relevance', description='Relevance with query', dimension=1),
    Component(name='diversity', description='Diversity between entities', dimension=2)
]

# Initialize
llm_evaluator = LLMEvaluator(mock_api=False)  # Set to True for testing
algorithm = PCSAlgorithm(components, llm_evaluator, budget_rate=5, epsilon=0.01)

# Run algorithm
entities = {...}  # Dictionary of Entity objects
final_package, metadata = algorithm.run(entities, k=3, query="your query")
```

### AQS Algorithm (Exact)

```python
from utils.models import Entity, Component
from utils.llm_interface import LLMEvaluator
from AQS.algorithm import AQSAlgorithm

# Define components
components = [
    Component(name='relevance', description='Relevance with query', dimension=1),
    Component(name='diversity', description='Diversity between entities', dimension=2)
]

# Initialize
llm_evaluator = LLMEvaluator(mock_api=False)  # Set to True for testing
algorithm = AQSAlgorithm(
    entities=entities,
    components=components,
    k=3,
    alpha=0.8,  # Confidence threshold
    query="your query",
    llm_evaluator=llm_evaluator
)

# Run algorithm
final_package, metadata = algorithm.run()
```

## Testing

### Test PCS Algorithm
```bash
# Test scoring function
python PCS/test_scoring_PCS.py

# Test full algorithm
python PCS/test_algorithm.py
```

### Test AQS Algorithm
```bash
# Test scoring function
python AQS/test_scoring.py

# Test package manager
python AQS/test_package_manager.py

# Test full algorithm
python AQS/test_algorithm.py
```

## Data Format

Entities can be loaded from CSV with columns: `entity_id`, `entity_name`, `data`

Example:
```csv
entity_id,entity_name,data
e1,Product A,Great product with excellent features
e2,Product B,Good quality and fast delivery
```

## Algorithm Details

### PCS (Approximate Algorithm)
- Starts with initial top-k package (by unary components)
- Iteratively swaps entities using Chebyshev's inequality to guide candidate selection
- Uses contribution scores to identify weakest entity
- Efficiently prunes candidates using tail probability estimates

### AQS (Exact Algorithm)
- Maintains a set of candidate packages with interval bounds
- Uses admissible heuristic to select most informative questions
- Updates bounds as questions are answered
- Prunes dominated packages based on bounds
- Terminates when alpha-top condition is satisfied

## Parameters

### PCS Parameters
- `budget_rate` (s): Number of top candidates to evaluate exactly per iteration
- `epsilon`: Convergence threshold for tail probability

### AQS Parameters
- `alpha`: Confidence threshold [0, 1] for alpha-top package guarantee
- `k`: Package size
- `query`: User query string

## Notes

- Component values are always returned as intervals [lower, upper] in range [0, 1]
- Use `mock_api=True` in LLMEvaluator for testing without API calls
- All component values are cached to avoid redundant LLM calls
- Package scores are computed as intervals, even when all component values are known
