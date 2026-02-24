"""
Generate MGT CSVs for data/yelp_dataset.csv.
Run from project root: python run_mgt/run_MGT_yelp.py --scoring f5|f6 --n_entities N --uniform_every_k K
Uses images_base_path so dimension-1 components can send entity images to the LLM when image_id is set.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path
from utils.llm_interface import LLMEvaluator
from preprocessing.MGT import MGT


def main():
    parser = argparse.ArgumentParser(description="Generate MGT for Yelp dataset")
    parser.add_argument("--scoring", choices=["f5", "f6"], required=True, help="Scoring function: f5 or f6")
    parser.add_argument("--n_entities", type=int, required=True, help="Number of entities (first n)")
    parser.add_argument("--uniform_every_k", type=int, default=1, help="Do real LLM call every k rows (default: 1)")
    args = parser.parse_args()

    csv_path = get_dataset_path(args.scoring)
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    config = get_config(args.scoring)
    query = config["query"]
    components = config["components"]

    photos_dir = str(ROOT / "data" / "yelp_dataset" / "photos")
    llm_evaluator = LLMEvaluator(
        mock_api=False,
        use_MGT=False,
        images_base_path=photos_dir,
    )
    output_dir = ROOT / "mgt_Results"
    mgt = MGT(entities_csv_path=str(csv_path), components=components, uniform_every_k=args.uniform_every_k)

    print(f"Generating MGT (yelp) scoring={args.scoring} for {csv_path}")
    print(f"Query: {query}")
    print(f"Output dir: {output_dir}  |  n_entities: {args.n_entities}  |  uniform_every_k: {args.uniform_every_k}")
    print(f"Images: {photos_dir} (used for dim-1 when entity has image_id)")
    paths = mgt.generate(llm_evaluator, query, output_dir=str(output_dir), n=args.n_entities)
    print("Done. Written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


# python run_mgt/run_MGT_yelp.py --scoring f5 --n_entities 10000 --uniform_every_k 10000
# python run_mgt/run_MGT_yelp.py --scoring f6 --n_entities 10000 --uniform_every_k 10000

if __name__ == "__main__":
    main()
