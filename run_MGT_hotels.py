"""
Generate MGT CSVs for data/hotels_dataset.csv.
Run: python run_MGT_hotels.py --scoring f1|f2 --n_entities N --uniform_every_k K
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.models import Component
from utils.llm_interface import LLMEvaluator
from preprocessing.MGT import MGT


F1 = {
    "query": "Find hotels where each hotel balances comfort, safety, and accessibility, and that collectively span different price tiers.",
    "components": [
        Component(name="c1", description="Hotels that each balances comfort, safety, and accessibility", dimension=1),
        Component(name="c2", description="Hotels spanning different price tiers", dimension=2),
    ],
}

F2 = {
    "query": "Find hotels near cultural and academic hubs that offer a quiet, work-friendly environment and are close to public transportation, while ensuring a diverse mix of large hotel brands, boutique hotels, and local properties.",
    "components": [
        Component(name="c1", description="Hotels that are near cultural and academic hubs, and offer a quiet work-friendly environment", dimension=1),
        Component(name="c2", description="Hotels that are close to public transportation", dimension=1),
        Component(name="c3", description="Diverse hotels, ensuring a mix of large hotel brands, boutique hotels, and local properties", dimension=2),
    ],
}

CONFIGS = {"f1": F1, "f2": F2}


def main():
    parser = argparse.ArgumentParser(description="Generate MGT for hotels dataset")
    parser.add_argument("--scoring", choices=["f1", "f2"], required=True, help="Scoring function: f1 or f2")
    parser.add_argument("--n_entities", type=int, required=True, help="Number of entities (first n)")
    parser.add_argument("--uniform_every_k", type=int, default=1, help="Do real LLM call every k rows (default: 1)")
    args = parser.parse_args()

    csv_path = ROOT / "data" / "hotels_dataset.csv"
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    config = CONFIGS[args.scoring]
    query = config["query"]
    components = config["components"]

    llm_evaluator = LLMEvaluator(mock_api=False, use_MGT=False)
    output_dir = ROOT / "mgt_Results"
    mgt = MGT(entities_csv_path=str(csv_path), components=components, uniform_every_k=args.uniform_every_k)

    print(f"Generating MGT (hotels) scoring={args.scoring} for {csv_path}")
    print(f"Query: {query}")
    print(f"Output dir: {output_dir}  |  n_entities: {args.n_entities}  |  uniform_every_k: {args.uniform_every_k}")
    paths = mgt.generate(llm_evaluator, query, output_dir=str(output_dir), n=args.n_entities)
    print("Done. Written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


# python run_MGT_hotels.py --scoring f1 --n_entities 10000 --uniform_every_k 10000
# python run_MGT_hotels.py --scoring f2 --n_entities 10000 --uniform_every_k 10000

if __name__ == "__main__":
    main()
