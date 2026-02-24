"""
Scalability experiment (exact algorithm): fix k, increase number of packages.
Measures time_maintain_packages, time_ask_next_question, time_process_response, time_total.
Uses mock_api=True. Output CSV: scalability_exact_increase_packages_k<k>_<F?>.csv
Run from project root: python experiments/scalability/run_scalability_increase_packages.py --scoring f1 [--k 2]
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
from preprocessing.load_data import load_entities_from_csv
from AQS.algorithm import AQSAlgorithm

# Package counts to sweep (exact algorithm: we choose n so that C(n,k) is close to these)
#PACKAGE_COUNTS = [1000, 2000, 5000, 10000, 20000]
PACKAGE_COUNTS = [10, 20, 50, 100, 200]
FIXED_K = 2
ALPHA = 0.5


def n_entities_for_package_count(package_count: int, k: int) -> int:
    """Minimum n such that C(n, k) >= package_count."""
    if k == 2:
        # n*(n-1)/2 >= P  =>  n >= (1 + sqrt(1+8*P))/2
        n = math.ceil((1 + math.sqrt(1 + 8 * package_count)) / 2)
        return n
    # General: binary search or approximate
    low, high = k, 2000
    while high - low > 1:
        mid = (low + high) // 2
        num_pkgs = math.comb(mid, k) if mid >= k else 0
        if num_pkgs >= package_count:
            high = mid
        else:
            low = mid
    return high


def entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def main():
    parser = argparse.ArgumentParser(description="Scalability: exact algo, increase packages")
    parser.add_argument("--scoring", choices=list(CONFIGS.keys()), required=True, help="Scoring function (f1–f6)")
    parser.add_argument("--k", type=int, default=FIXED_K, help="Fixed package size k")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: scalability_outputs/scalability_outputs)")
    args = parser.parse_args()

    scoring = args.scoring.lower()
    k = args.k
    csv_path = get_dataset_path(scoring)
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    config = get_config(scoring)
    query = config["query"]
    components = config["components"]

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "scalability_outputs" / "scalability_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"scalability_exact_increase_packages_k{k}_{scoring}_{ts}.csv"

    print(f"Scalability (exact): scoring={scoring}, k={k}, package_counts={PACKAGE_COUNTS}")
    print(f"Output: {out_file}")

    rows = []
    for target_packages in PACKAGE_COUNTS:
        n = n_entities_for_package_count(target_packages, k)
        entities_all = load_entities_from_csv(str(csv_path))
        entity_ids_sorted = sorted(entities_all.keys(), key=entity_order_key)[:n]
        if len(entity_ids_sorted) < n:
            print(f"  Skip target_packages={target_packages}: only {len(entity_ids_sorted)} entities in CSV")
            continue
        entities = {eid: entities_all[eid] for eid in entity_ids_sorted}
        num_packages = math.comb(len(entities), k)
        print(f"  n_entities={n} -> #packages={num_packages} (target {target_packages})")

        llm_evaluator = LLMEvaluator(mock_api=True, use_MGT=False)
        algorithm = AQSAlgorithm(
            entities=entities,
            components=components,
            k=k,
            alpha=ALPHA,
            query=query,
            llm_evaluator=llm_evaluator,
            initial_packages=None,
            print_log=False,
            init_dim_1=True,
            is_next_q_random=False,
            return_timings=True,
        )
        final_package, metadata = algorithm.run()

        rows.append({
            "#packages": num_packages,
            "time_maintain_packages": metadata.get("time_maintain_packages", 0),
            "time_ask_next_question": metadata.get("time_ask_next_question", 0),
            "time_process_response": metadata.get("time_process_response", 0),
            "time_total": metadata.get("time_total", 0),
        })

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["#packages", "time_maintain_packages", "time_ask_next_question", "time_process_response", "time_total"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_file}")


# python experiments/scalability/run_scalability_increase_packages.py --scoring f1
# python experiments/scalability/run_scalability_increase_packages.py --scoring f2
# python experiments/scalability/run_scalability_increase_packages.py --scoring f3
# python experiments/scalability/run_scalability_increase_packages.py --scoring f4
# python experiments/scalability/run_scalability_increase_packages.py --scoring f5
# python experiments/scalability/run_scalability_increase_packages.py --scoring f6

if __name__ == "__main__":
    main()
