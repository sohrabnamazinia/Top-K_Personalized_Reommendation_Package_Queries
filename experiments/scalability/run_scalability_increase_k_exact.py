"""
Scalability experiment (exact algorithm): fix #packages (1000), increase k.
k = 2, 4, 6, 8, 10. For each k we build 1000 packages of size k from n entities (n chosen so
C(n,10) >= 1000), run exact AQS with those as initial_packages, measure same 4 timings.
Uses mock_api=True. Output CSV with timestamp.
Run from project root: python experiments/scalability/run_scalability_increase_k_exact.py --scoring f1
"""
import argparse
import csv
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path, CONFIGS
from utils.llm_interface import LLMEvaluator
from utils.models import Package
from preprocessing.load_data import load_entities_from_csv
from AQS.algorithm import AQSAlgorithm

K_VALUES = [2, 4, 6, 8, 10]
TARGET_PACKAGES = 1000
ALPHA = 0.5


def n_entities_for_package_count(package_count: int, k: int) -> int:
    """Minimum n such that C(n, k) >= package_count."""
    if k == 2:
        n = math.ceil((1 + math.sqrt(1 + 8 * package_count)) / 2)
        return n
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


def sample_packages(entity_ids: list, k: int, count: int) -> list:
    """Sample `count` packages of size k from entity_ids (without replacement)."""
    combos = list(combinations(entity_ids, k))
    if len(combos) <= count:
        return [Package(entities=set(c)) for c in combos]
    chosen = random.sample(combos, count)
    return [Package(entities=set(c)) for c in chosen]


def main():
    parser = argparse.ArgumentParser(description="Scalability: exact algo, increase k")
    parser.add_argument("--scoring", choices=list(CONFIGS.keys()), required=True, help="Scoring function (f1–f6)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: scalability_outputs/scalability_outputs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling packages")
    args = parser.parse_args()

    random.seed(args.seed)
    scoring = args.scoring.lower()
    csv_path = get_dataset_path(scoring)
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    config = get_config(scoring)
    query = config["query"]
    components = config["components"]

    # n such that we have >= TARGET_PACKAGES for k=10 (most restrictive)
    n = n_entities_for_package_count(TARGET_PACKAGES, 10)
    entities_all = load_entities_from_csv(str(csv_path))
    entity_ids_sorted = sorted(entities_all.keys(), key=entity_order_key)[:n]
    if len(entity_ids_sorted) < n:
        print(f"Need at least {n} entities; CSV has {len(entity_ids_sorted)}")
        sys.exit(1)
    entities = {eid: entities_all[eid] for eid in entity_ids_sorted}
    entity_ids = entity_ids_sorted

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "scalability_outputs" / "scalability_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"scalability_exact_increase_k_{scoring}_{ts}.csv"

    print(f"Scalability (exact, increase k): scoring={scoring}, n_entities={len(entities)}, target_packages={TARGET_PACKAGES}, k_values={K_VALUES}")
    print(f"Output: {out_file}")

    rows = []
    for k in K_VALUES:
        if math.comb(len(entity_ids), k) < TARGET_PACKAGES:
            print(f"  Skip k={k}: C({len(entity_ids)},{k}) < {TARGET_PACKAGES}")
            continue
        initial_packages = sample_packages(entity_ids, k, TARGET_PACKAGES)
        num_packages = len(initial_packages)
        print(f"  k={k} -> #packages={num_packages}")

        llm_evaluator = LLMEvaluator(mock_api=True, use_MGT=False)
        algorithm = AQSAlgorithm(
            entities=entities,
            components=components,
            k=k,
            alpha=ALPHA,
            query=query,
            llm_evaluator=llm_evaluator,
            initial_packages=initial_packages,
            print_log=False,
            init_dim_1=True,
            is_next_q_random=False,
            return_timings=True,
        )
        final_package, metadata = algorithm.run()
        initial_n = metadata.get("initial_package_count", num_packages)
        final_n = metadata.get("final_package_count", 0)
        pct = (100.0 * (initial_n - final_n) / initial_n) if initial_n else 0.0
        print(f"  -> {pct:.1f}% of packages pruned")

        rows.append({
            "k": k,
            "#packages": num_packages,
            "time_maintain_packages": metadata.get("time_maintain_packages", 0),
            "time_ask_next_question": metadata.get("time_ask_next_question", 0),
            "time_process_response": metadata.get("time_process_response", 0),
            "time_total": metadata.get("time_total", 0),
        })

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["k", "#packages", "time_maintain_packages", "time_ask_next_question", "time_process_response", "time_total"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_file}")


# python experiments/scalability/run_scalability_increase_k_exact.py --scoring f1
# python experiments/scalability/run_scalability_increase_k_exact.py --scoring f2
# python experiments/scalability/run_scalability_increase_k_exact.py --scoring f3
# python experiments/scalability/run_scalability_increase_k_exact.py --scoring f4
# python experiments/scalability/run_scalability_increase_k_exact.py --scoring f5
# python experiments/scalability/run_scalability_increase_k_exact.py --scoring f6

if __name__ == "__main__":
    main()
