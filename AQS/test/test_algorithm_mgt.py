"""AQS algorithm test - MGT (read scores from pre-generated MGT CSVs)."""
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

# ---------- Input parameters (MGT run) ----------
ROOT = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / "data" / "sample_data.csv"
OUTPUT_DIR = str(ROOT / "mgt_Results")
N_ENTITIES = 10
K = 3
ALPHA = 0.8
QUERY = "Find products with good quality"
PRINT_LOG = True
SELECTION_STRATEGY = "aqs"
HEURISTIC_TOP_PACKAGES_PCT = 0.01  # Set to None to disable heuristic pruning of top packages after each iteration
# -----------------------------------------------

COMPONENTS = [
    Component(name="relevance", description="Relevance of the entity with the user query", dimension=1),
    Component(name="diversity", description="Diversity between two entities", dimension=2),
]


def test_algorithm():
    entities_all = load_entities_from_csv(str(CSV_PATH))
    entities = dict(list(entities_all.items())[:N_ENTITIES]) if N_ENTITIES else dict(entities_all)

    llm_evaluator = LLMEvaluator(
        use_MGT=True,
        entities_csv_path=str(CSV_PATH),
        components=COMPONENTS,
        output_dir=OUTPUT_DIR,
        n=N_ENTITIES if N_ENTITIES is not None else len(entities),
    )
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
    print("AQS Algorithm (MGT)")
    print("=" * 60)
    print("Entities:", len(entities), "MGT dir:", OUTPUT_DIR)
    print("k:", K, "alpha:", ALPHA, "query:", QUERY)
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
        print("Bounds:", lb, ub)
    print("Questions asked:", metadata["questions_asked"], "iterations:", len(metadata["iterations"]))

    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    log_path = os.path.join(outputs_dir, "aqs_mgt_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with open(log_path, "w") as f:
        json.dump({"parameters": {"k": K, "alpha": ALPHA, "query": QUERY, "n_entities": len(entities), "output_dir": OUTPUT_DIR}, "result": metadata}, f, indent=2)
    print("Log:", log_path)

    summary_dir = os.path.join(os.path.dirname(__file__), "outputs_summary")
    os.makedirs(summary_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_bounds = algorithm.package_manager.get_bounds(final_package) if final_package else (None, None)
    with open(os.path.join(summary_dir, f"aqs_mgt_{ts}.txt"), "w") as f:
        f.write(f"AQS Algorithm (MGT) - {datetime.now().isoformat()}\n")
        f.write(f"n_entities={len(entities)} k={K} alpha={ALPHA} query={QUERY} output_dir={OUTPUT_DIR}\n\n")
        f.write(f"Exact baseline:\n  best_package: {sorted(baseline_package.entities)}\n  top5_scores: {baseline_scores[:5]}\n\n")
        f.write(f"Algorithm:\n  final_package: {list(final_package.entities) if final_package else None}\n  bounds: {final_bounds}\n  questions_asked: {metadata['questions_asked']} iterations: {len(metadata['iterations'])}\n")


if __name__ == "__main__":
    test_algorithm()
