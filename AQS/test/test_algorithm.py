"""AQS algorithm test – regular (LLM or mock, no MGT)."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.models import Component
from utils.llm_interface import LLMEvaluator
from preprocessing.load_data import load_entities_from_csv
from BASELINE.algorithm import ExactBaseline
from AQS.algorithm import AQSAlgorithm

# ---------- Input parameters (regular run: no MGT) ----------
ROOT = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / "data" / "sample_data.csv"
N_ENTITIES = 10      # None = use all from CSV; else int (e.g. 10) to take first N
K = 3
ALPHA = 0.8
QUERY = "Find products with good quality"
MOCK_API = True           # True = no API key; False = use OpenAI
PRINT_LOG = True
SELECTION_STRATEGY = "aqs"
HEURISTIC_TOP_PACKAGES_PCT = 0.01  # Set to None to disable heuristic pruning of top packages after each iteration
# -----------------------------------------------------------

COMPONENTS = [
    Component(name="relevance", description="Relevance of the entity with the user query", dimension=1),
    Component(name="diversity", description="Diversity between two entities", dimension=2),
]


def test_algorithm():
    entities_all = load_entities_from_csv(str(CSV_PATH))
    entities = dict(list(entities_all.items())[:N_ENTITIES]) if N_ENTITIES else dict(entities_all)

    llm_evaluator = LLMEvaluator(mock_api=MOCK_API, use_MGT=False)
    baseline = ExactBaseline(COMPONENTS, llm_evaluator)
    algorithm = AQSAlgorithm(
        entities=entities,
        components=COMPONENTS,
        k=K,
        alpha=ALPHA,
        query=QUERY,
        llm_evaluator=llm_evaluator,
        initial_packages=None,
        print_log=PRINT_LOG,
        selection_strategy=SELECTION_STRATEGY,
        heuristic_top_packages_pct=HEURISTIC_TOP_PACKAGES_PCT
    )
    print("=" * 60)
    print("AQS Algorithm (regular)")
    print("=" * 60)
    print(f"Entities: {len(entities)}")
    print(f"k: {K}, alpha: {ALPHA}, query: {QUERY}")
    print()

    baseline_package, baseline_scores = baseline.run(entities, QUERY, K)
    print("Baseline best package:", sorted(baseline_package.entities))
    print("Baseline top-5 scores:", baseline_scores[:5])
    print()

    print("Running AQS...")
    final_package, metadata = algorithm.run()
    print()
    print("Final package:", list(final_package.entities) if final_package else None)
    if final_package:
        lb, ub = algorithm.package_manager.get_bounds(final_package)
        print(f"Bounds: [{lb}, {ub}]")
    print(f"Questions asked: {metadata['questions_asked']}, iterations: {len(metadata['iterations'])}")

    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    log_path = os.path.join(outputs_dir, f"aqs_regular_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w") as f:
        json.dump({"parameters": {"k": K, "alpha": ALPHA, "query": QUERY, "n_entities": len(entities)}, "result": metadata}, f, indent=2)
    print(f"Log: {log_path}")

    summary_dir = os.path.join(os.path.dirname(__file__), "outputs_summary")
    os.makedirs(summary_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_bounds = algorithm.package_manager.get_bounds(final_package) if final_package else (None, None)
    with open(os.path.join(summary_dir, f"aqs_regular_{ts}.txt"), "w") as f:
        f.write(f"AQS Algorithm (regular) - {datetime.now().isoformat()}\n")
        f.write(f"n_entities={len(entities)} k={K} alpha={ALPHA} query={QUERY}\n\n")
        f.write(f"Exact baseline:\n  best_package: {sorted(baseline_package.entities)}\n  top5_scores: {baseline_scores[:5]}\n\n")
        f.write(f"Algorithm:\n  final_package: {list(final_package.entities) if final_package else None}\n  bounds: {final_bounds}\n  questions_asked: {metadata['questions_asked']} iterations: {len(metadata['iterations'])}\n")


if __name__ == "__main__":
    test_algorithm()
