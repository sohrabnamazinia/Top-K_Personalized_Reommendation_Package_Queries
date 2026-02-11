"""
One-time script to generate MGT CSVs for data/sample_data.csv.
Run from project root: python run_MGT.py

Uses mock_api=True by default (no API key). Set mock_api=False to use real LLM.
"""
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.models import Component
from utils.llm_interface import LLMEvaluator
from preprocessing.MGT import MGT


def main():
    csv_path = ROOT / "data" / "sample_data.csv"
    query = "Find products with good quality"
    uniform_every_k = 20  # Set to None to disable uniform sampling

    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    components = [
        Component(
            name="relevance",
            description="Relevance of the entity with the user query",
            dimension=1,
        ),
        Component(
            name="diversity",
            description="Diversity between two entities",
            dimension=2,
        ),
    ]
    
    # use_MGT=False so evaluator can call LLM; generate() will use _evaluate_component_llm directly
    llm_evaluator = LLMEvaluator(
        mock_api=False,  # Set to False for real LLM (requires OPENAI_API_KEY)
        use_MGT=False,
    )
    output_dir = ROOT / "mgt_Results"
    mgt = MGT(entities_csv_path=str(csv_path), components=components, uniform_every_k=uniform_every_k)

    print(f"Generating MGT for {csv_path} (query: {query})")
    print(f"Output dir: {output_dir}")
    paths = mgt.generate(llm_evaluator, query, output_dir=str(output_dir))
    print("Done. Written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
