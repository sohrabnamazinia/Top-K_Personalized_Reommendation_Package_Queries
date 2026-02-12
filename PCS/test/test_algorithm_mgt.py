"""PCS algorithm test – MGT (read scores from pre-generated MGT CSVs)."""
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
from PCS.algorithm import PCSAlgorithm

# ---------- Input parameters (MGT run) ----------
ROOT = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / "data" / "sample_data.csv"
OUTPUT_DIR = str(ROOT / "mgt_Results")   # Directory where MGT_*.csv files live
N_ENTITIES = None         # None = use all from CSV; else int (must match an existing MGT n)
K = 3
QUERY = "Find products with good quality"
EPSILON = 0
SMART_INITIAL_PACKAGE = True
EXCEED_NUMBER_OF_CHANCE = 5
BUDGET_RATE = 10
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
    algorithm = PCSAlgorithm(
        components=COMPONENTS,
        llm_evaluator=llm_evaluator,
        budget_rate=BUDGET_RATE,
        epsilon=EPSILON,
        smart_initial_package=SMART_INITIAL_PACKAGE,
        exceed_number_of_chance=EXCEED_NUMBER_OF_CHANCE,
    )

    print("=" * 60)
    print("PCS Algorithm (MGT)")
    print("=" * 60)
    print(f"Entities: {len(entities)}, MGT dir: {OUTPUT_DIR}")
    print(f"k: {K}, query: {QUERY}, budget_rate: {BUDGET_RATE}, epsilon: {EPSILON}")
    print()

    baseline_package, baseline_scores = baseline.run(entities, QUERY, K)
    print("Baseline best package:", sorted(baseline_package.entities))
    print("Baseline top-5 scores:", baseline_scores[:5])
    print()

    final_package, metadata = algorithm.run(entities, K, QUERY)
    print("Final package:", list(final_package.entities))
    print("Final score:", metadata["final_score"])
    print("Iterations:", len(metadata["iterations"]))

    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(outputs_dir, f"pcs_mgt_{ts}.json")
    with open(log_path, "w") as f:
        json.dump({
            "parameters": {"k": K, "query": QUERY, "n_entities": len(entities), "output_dir": OUTPUT_DIR, "budget_rate": BUDGET_RATE, "epsilon": EPSILON},
            "result": {"final_package": list(final_package.entities), "final_score": metadata["final_score"], "iterations": metadata["iterations"]},
        }, f, indent=2)
    print(f"Log: {log_path}")

    summary_dir = os.path.join(os.path.dirname(__file__), "outputs_summary")
    os.makedirs(summary_dir, exist_ok=True)
    stop_reason = metadata["iterations"][-1]["stop_reason"] if metadata["iterations"] else None
    summary = (
        f"PCS Algorithm (MGT) - {datetime.now().isoformat()}\n"
        f"n_entities={len(entities)} k={K} query={QUERY} output_dir={OUTPUT_DIR} budget_rate={BUDGET_RATE} epsilon={EPSILON}\n\n"
        f"Exact baseline:\n  best_package: {sorted(baseline_package.entities)}\n  top5_scores: {baseline_scores[:5]}\n\n"
        f"Algorithm:\n  final_package: {list(final_package.entities)}\n  final_score: {metadata['final_score']}\n  iterations: {len(metadata['iterations'])}\n  stop_reason: {stop_reason}\n"
    )
    with open(os.path.join(summary_dir, f"pcs_mgt_{ts}.txt"), "w") as f:
        f.write(summary)

    return final_package, metadata


if __name__ == "__main__":
    test_algorithm()
