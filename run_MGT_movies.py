"""
Generate MGT CSVs for data/movies_dataset.csv.
Run: python run_MGT_movies.py --scoring f3|f4 --n_entities N --uniform_every_k K
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.models import Component
from utils.llm_interface import LLMEvaluator
from preprocessing.MGT import MGT


F3 = {
    "query": "package of high-quality movies about space exploration and astronauts that feel realistic and scientifically grounded. The package should be diverse in tone and perspective (e.g., survival-focused, philosophical, political, emotional), so the movies don't all tell the same kind of story.",
    "components": [
        Component(name="c1", description="High quality movies about space exploration and astronauts with realistic or science-driven themes.", dimension=1),
        Component(name="c2", description="Diverse in tone and perspective (e.g., survival-focused, philosophical, political, emotional), so the movies don't all tell the same kind of story.", dimension=2),
    ],
}

F4 = {
    "query": "Intellectually engaging and character-driven movies that could be watched with family during holidays. The movies should have diverse cultural perspectives and length",
    "components": [
        Component(name="c1", description="Intellectually engaging movies", dimension=1),
        Component(name="c2", description="Character-driven movies", dimension=1),
        Component(name="c3", description="Movies that could be watched with family during holidays", dimension=1),
        Component(name="c4", description="Movies with diverse cultural perspectives and length", dimension=2),
    ],
}

CONFIGS = {"f3": F3, "f4": F4}


def main():
    parser = argparse.ArgumentParser(description="Generate MGT for movies dataset")
    parser.add_argument("--scoring", choices=["f3", "f4"], required=True, help="Scoring function: f3 or f4")
    parser.add_argument("--n_entities", type=int, required=True, help="Number of entities (first n)")
    parser.add_argument("--uniform_every_k", type=int, default=1, help="Do real LLM call every k rows (default: 1)")
    args = parser.parse_args()

    csv_path = ROOT / "data" / "movies_dataset.csv"
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    config = CONFIGS[args.scoring]
    query = config["query"]
    components = config["components"]

    llm_evaluator = LLMEvaluator(mock_api=False, use_MGT=False)
    output_dir = ROOT / "mgt_Results"
    mgt = MGT(entities_csv_path=str(csv_path), components=components, uniform_every_k=args.uniform_every_k)

    print(f"Generating MGT (movies) scoring={args.scoring} for {csv_path}")
    print(f"Query: {query}")
    print(f"Output dir: {output_dir}  |  n_entities: {args.n_entities}  |  uniform_every_k: {args.uniform_every_k}")
    paths = mgt.generate(llm_evaluator, query, output_dir=str(output_dir), n=args.n_entities)
    print("Done. Written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


# python run_MGT_movies.py --scoring f3 --n_entities 10000 --uniform_every_k 10000
# python run_MGT_movies.py --scoring f4 --n_entities 10000 --uniform_every_k 10000

if __name__ == "__main__":
    main()
