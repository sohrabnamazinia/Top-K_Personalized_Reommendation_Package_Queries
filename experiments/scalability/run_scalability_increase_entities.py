"""
Scalability experiment: vary #entities. Measures time_maintain_packages, time_ask_next_question,
time_process_response, time_total for AQS or PCS. Output: experiments_outputs/scalability_outputs/.
With --synthetic, generates synthetic entities (no CSV) so you can scale to 1M+ entities.
Run: python experiments/scalability/run_scalability_increase_entities.py --scoring f1 --method aqs
     python experiments/scalability/run_scalability_increase_entities.py --scoring f1 --method pcs --synthetic
"""
import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path, CONFIGS
from utils.llm_interface import LLMEvaluator
from utils.models import Entity
from preprocessing.load_data import load_entities_from_csv
from AQS.algorithm import AQSAlgorithm
from PCS.algorithm import PCSAlgorithm

# With --synthetic: scale to 1M+ (no CSV). Without: limited by CSV size (~20–30k).
ENTITY_COUNTS_SYNTHETIC = [50]
ENTITY_COUNTS_CSV = [1000, 5000, 10000, 15000, 20000]  # typical real-data sizes
FIXED_K = 3
ALPHA = 0.8


def generate_synthetic_entities(n: int) -> dict:
    """Return a dict of n synthetic entities (e1..eN) with minimal placeholder data."""
    entities = {}
    for i in range(1, n + 1):
        eid = f"e{i}"
        entities[eid] = Entity(
            id=eid,
            name=f"Entity {i}",
            data=f"Synthetic entity {i}.",
            image_id=None,
        )
    return entities


def entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def main():
    parser = argparse.ArgumentParser(description="Scalability: increase #entities")
    parser.add_argument("--scoring", choices=list(CONFIGS.keys()), required=True, help="Scoring (f1–f6)")
    parser.add_argument("--method", choices=["aqs", "pcs"], required=True, help="Algorithm: aqs or pcs")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic entities (no CSV); scale to 1M+")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (default: experiments_outputs/scalability_outputs)")
    args = parser.parse_args()

    scoring = args.scoring.lower()
    method = args.method.lower()
    use_synthetic = args.synthetic

    config = get_config(scoring)
    query = config["query"]
    components = config["components"]

    if use_synthetic:
        entity_counts = ENTITY_COUNTS_SYNTHETIC
        entities_all = None
        entity_ids_sorted = None
        print(f"Scalability (increase entities, SYNTHETIC): method={method}, scoring={scoring}, entity_counts={entity_counts}, k={FIXED_K}, alpha={ALPHA}")
    else:
        csv_path = get_dataset_path(scoring)
        if not csv_path.exists():
            print(f"Dataset not found: {csv_path}")
            sys.exit(1)
        entities_all = load_entities_from_csv(str(csv_path))
        entity_ids_sorted = sorted(entities_all.keys(), key=entity_order_key)
        max_n = len(entity_ids_sorted)
        if max_n < max(ENTITY_COUNTS_CSV):
            entity_counts = [n for n in ENTITY_COUNTS_CSV if n <= max_n]
            if not entity_counts:
                entity_counts = [max_n]
        else:
            entity_counts = ENTITY_COUNTS_CSV
        print(f"Scalability (increase entities, CSV): method={method}, scoring={scoring}, entity_counts={entity_counts}, k={FIXED_K}, alpha={ALPHA}")

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "experiments_outputs" / "scalability_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "synthetic" if use_synthetic else scoring
    out_file = output_dir / f"scalability_increase_entities_{method}_{suffix}_{ts}.csv"
    print(f"Output: {out_file}")

    rows = []
    for n in entity_counts:
        if use_synthetic:
            entities = generate_synthetic_entities(n)
        else:
            entity_ids = entity_ids_sorted[:n]
            entities = {eid: entities_all[eid] for eid in entity_ids}
        num_packages = math.comb(len(entities), FIXED_K) if method == "aqs" else 1
        print(f"  n_entities={n} -> #packages={num_packages}")

        llm_evaluator = LLMEvaluator(mock_api=True, use_MGT=False)
        if method == "aqs":
            algorithm = AQSAlgorithm(
                entities=entities,
                components=components,
                k=FIXED_K,
                alpha=ALPHA,
                query=query,
                llm_evaluator=llm_evaluator,
                initial_packages=None,
                print_log=False,
                init_dim_1=True,
                return_timings=True,
            )
            final_package, metadata = algorithm.run()
        else:
            algorithm = PCSAlgorithm(
                components=components,
                llm_evaluator=llm_evaluator,
                smart_initial_package=False,
                return_timings=True,
            )
            final_package, metadata = algorithm.run(entities, FIXED_K, query)

        initial_n = metadata.get("initial_package_count", num_packages)
        final_n = metadata.get("final_package_count", 0)
        if method == "aqs" and initial_n:
            pct = 100.0 * (initial_n - final_n) / initial_n
            print(f"  -> {pct:.1f}% of packages pruned")

        rows.append({
            "method": method,
            "#entities": n,
            "#packages": num_packages,
            "time_maintain_packages": metadata.get("time_maintain_packages", 0),
            "time_ask_next_question": metadata.get("time_ask_next_question", 0),
            "time_process_response": metadata.get("time_process_response", 0),
            "time_total": metadata.get("time_total", 0),
        })

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "#entities", "#packages", "time_maintain_packages", "time_ask_next_question", "time_process_response", "time_total"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_file}")


# CSV (real data, ~20–30k max):
#   python experiments/scalability/run_scalability_increase_entities.py --scoring f1 --method aqs
#   python experiments/scalability/run_scalability_increase_entities.py --scoring f1 --method pcs
# Synthetic (scale to 1M, time-only):
#   python experiments/scalability/run_scalability_increase_entities.py --scoring f1 --method aqs --synthetic
#   python experiments/scalability/run_scalability_increase_entities.py --scoring f1 --method pcs --synthetic

if __name__ == "__main__":
    main()
