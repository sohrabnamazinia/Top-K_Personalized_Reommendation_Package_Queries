"""
Scalability experiment: vary k (AQS vs PCS).

We measure:
  - time_maintain_packages
  - time_ask_next_question
  - time_process_response
  - time_total

Settings:
  - #entities fixed (default 100)
  - alpha fixed (default 0.8)
  - k swept over {2,4,6,8}

Important: For AQS, enumerating all packages is infeasible when k grows.
This script samples a fixed number of initial packages for AQS (default 5000) so the run is tractable.

Run from project root, e.g.:
  python experiments/scalability/run_scalability_increase_k.py --scoring f1 --method pcs
  python experiments/scalability/run_scalability_increase_k.py --scoring f1 --method aqs --aqs_package_limit 5000
"""

import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path, CONFIGS
from utils.llm_interface import LLMEvaluator
from utils.models import Entity, Package
from preprocessing.load_data import load_entities_from_csv
from AQS.algorithm import AQSAlgorithm
from PCS.algorithm import PCSAlgorithm

K_VALUES_DEFAULT = [2, 4, 6, 8, 10]
DEFAULT_N_ENTITIES = 100000
ALPHA = 0.8


def generate_synthetic_entities(n: int) -> dict:
    entities = {}
    for i in range(1, n + 1):
        eid = f"e{i}"
        entities[eid] = Entity(id=eid, name=f"Entity {i}", data=f"Synthetic entity {i}.", image_id=None)
    return entities


def entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def sample_initial_packages(entity_ids: list[str], k: int, count: int, seed: int) -> list[Package]:
    """Sample `count` unique packages of size k (approx uniform) without enumerating combinations."""
    rng = random.Random(seed + 31 * k)
    seen: set[frozenset[str]] = set()
    max_attempts = max(10_000, count * 20)
    attempts = 0
    while len(seen) < count and attempts < max_attempts:
        attempts += 1
        pkg = frozenset(rng.sample(entity_ids, k))
        seen.add(pkg)
    return [Package(entities=set(s)) for s in seen]


def parse_k_list(s: str) -> list[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(int(p))
    return out


def main():
    parser = argparse.ArgumentParser(description="Scalability: increase k (AQS vs PCS)")
    parser.add_argument("--scoring", choices=list(CONFIGS.keys()), required=True, help="Scoring (f1–f6)")
    parser.add_argument("--method", choices=["aqs", "pcs"], required=True, help="Algorithm: aqs or pcs")
    parser.add_argument("--n_entities", type=int, default=DEFAULT_N_ENTITIES, help="Number of entities (default: 100)")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Alpha (default: 0.8)")
    parser.add_argument("--k_values", type=str, default="2,4,6,8,10", help="Comma-separated k values (default: 2,4,6,8,10)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic entities (no CSV)")
    parser.add_argument("--aqs_package_limit", type=int, default=5000, help="AQS only: number of initial packages to sample (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (default: experiments_outputs/scalability_outputs)")
    args = parser.parse_args()

    scoring = args.scoring.lower()
    method = args.method.lower()
    n_entities = int(args.n_entities)
    alpha = float(args.alpha)
    k_values = parse_k_list(args.k_values) or K_VALUES_DEFAULT

    config = get_config(scoring)
    query = config["query"]
    components = config["components"]

    if args.synthetic:
        entities = generate_synthetic_entities(n_entities)
    else:
        csv_path = get_dataset_path(scoring)
        if not csv_path.exists():
            print(f"Dataset not found: {csv_path}")
            sys.exit(1)
        entities_all = load_entities_from_csv(str(csv_path))
        entity_ids_sorted = sorted(entities_all.keys(), key=entity_order_key)[:n_entities]
        if len(entity_ids_sorted) < n_entities:
            print(f"CSV has only {len(entity_ids_sorted)} entities; requested {n_entities}")
            sys.exit(1)
        entities = {eid: entities_all[eid] for eid in entity_ids_sorted}

    entity_ids = list(entities.keys())

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "experiments_outputs" / "scalability_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"scalability_increase_k_{method}_{scoring}_{ts}.csv"

    print(f"Scalability (increase k): method={method}, scoring={scoring}, n_entities={n_entities}, alpha={alpha}, k_values={k_values}")
    if method == "aqs":
        print(f"AQS initial package sample size: {args.aqs_package_limit}")
    print(f"Output: {out_file}")

    rows: list[dict] = []
    for k in k_values:
        if k > n_entities:
            print(f"  Skip k={k}: k > n_entities")
            continue

        llm_evaluator = LLMEvaluator(mock_api=True, use_MGT=False)
        if method == "aqs":
            initial_packages = sample_initial_packages(entity_ids, k, args.aqs_package_limit, seed=args.seed)
            algorithm = AQSAlgorithm(
                entities=entities,
                components=components,
                k=k,
                alpha=alpha,
                query=query,
                llm_evaluator=llm_evaluator,
                initial_packages=initial_packages,
                print_log=False,
                init_dim_1=True,
                is_next_q_random=False,
                return_timings=True,
            )
            _, metadata = algorithm.run()
        else:
            algorithm = PCSAlgorithm(
                components=components,
                llm_evaluator=llm_evaluator,
                return_timings=True,
            )
            _, metadata = algorithm.run(entities, k, query)

        rows.append(
            {
                "method": method,
                "#entities": n_entities,
                "k": k,
                "time_maintain_packages": metadata.get("time_maintain_packages", 0),
                "time_ask_next_question": metadata.get("time_ask_next_question", 0),
                "time_process_response": metadata.get("time_process_response", 0),
                "time_total": metadata.get("time_total", 0),
            }
        )
        print(f"  k={k}: total={rows[-1]['time_total']}")

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "#entities",
                "k",
                "time_maintain_packages",
                "time_ask_next_question",
                "time_process_response",
                "time_total",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
