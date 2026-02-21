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
    csv_path = ROOT / "data" / "movies_dataset.csv"
    query = "package of high-quality movies about space exploration and astronauts that feel realistic and scientifically grounded. The set should be diverse in tone and perspective (e.g., survival-focused, philosophical, political, emotional), so the movies don’t all tell the same kind of story."

    # Number of entities to use (first n after sorting by entity_id). None = use all in CSV.
    N_ENTITIES = 10000

    uniform_every_k = 10000 # Set to None to disable uniform sampling

    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    components = [
        Component(
            name="relevance",
            description="good highly rated hotels",
            dimension=1,
        ),
        Component(
            name="diversity",
            description="diverse in location",
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
    print(f"Output dir: {output_dir}  |  n_entities: {N_ENTITIES or 'all'}")
    paths = mgt.generate(llm_evaluator, query, output_dir=str(output_dir), n=N_ENTITIES)
    print("Done. Written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
