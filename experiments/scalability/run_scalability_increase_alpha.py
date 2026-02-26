"""
Scalability experiment: vary alpha (AQS vs PCS).

We measure:
  - time_maintain_packages
  - time_ask_next_question
  - time_process_response
  - time_total

Settings:
  - #entities fixed (default 100)
  - k fixed (default 3)
  - alpha swept over {0.1,0.3,0.5,0.7,0.9}

Notes:
  - For AQS, enumerating all packages can be expensive; we sample a fixed number of initial packages
    (default 5000) to keep runtime predictable.
  - PCS does not use alpha; we still run it per alpha so the CSV has consistent x-axis for plotting.

Run from project root, e.g.:
  python experiments/scalability/run_scalability_increase_alpha.py --scoring f1 --method aqs --synthetic
  python experiments/scalability/run_scalability_increase_alpha.py --scoring f1 --method pcs --synthetic
"""

from __future__ import annotations

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

ALPHA_VALUES_DEFAULT = [0.001, 0.01, 0.1, 0.5, 0.9]
DEFAULT_N_ENTITIES = 50
DEFAULT_K = 3


def generate_synthetic_entities(n: int) -> dict[str, Entity]:
    entities: dict[str, Entity] = {}
    for i in range(1, n + 1):
        eid = f"e{i}"
        entities[eid] = Entity(id=eid, name=f"Entity {i}", data=f"Synthetic entity {i}.", image_id=None)
    return entities


def entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def sample_initial_packages(entity_ids: list[str], k: int, count: int, seed: int) -> list[Package]:
    """Sample `count` unique packages of size k (approx uniform), without enumerating combinations."""
    rng = random.Random(seed + 31 * k)
    seen: set[frozenset[str]] = set()
    max_attempts = max(10_000, count * 30)
    attempts = 0
    while len(seen) < count and attempts < max_attempts:
        attempts += 1
        seen.add(frozenset(rng.sample(entity_ids, k)))
    return [Package(entities=set(s)) for s in seen]


def parse_alpha_list(s: str) -> list[float]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        out.append(float(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalability: increase alpha (AQS vs PCS)")
    parser.add_argument("--scoring", choices=list(CONFIGS.keys()), required=True, help="Scoring (f1–f6)")
    parser.add_argument("--method", choices=["aqs", "pcs"], required=True, help="Algorithm: aqs or pcs")
    parser.add_argument("--n_entities", type=int, default=DEFAULT_N_ENTITIES, help="Number of entities (default: 100)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Package size k (default: 3)")
    parser.add_argument("--alpha_values", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated alpha values")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic entities (no CSV)")
    parser.add_argument("--aqs_package_limit", type=int, default=5000, help="AQS only: number of initial packages to sample (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (default: experiments_outputs/scalability_outputs)")
    args = parser.parse_args()

    scoring = args.scoring.lower()
    method = args.method.lower()
    n_entities = int(args.n_entities)
    k = int(args.k)
    alpha_values = parse_alpha_list(args.alpha_values) or list(ALPHA_VALUES_DEFAULT)

    if k > n_entities:
        raise SystemExit("k cannot be greater than n_entities")

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
    out_file = output_dir / f"scalability_increase_alpha_{method}_{scoring}_{ts}.csv"

    print(f"Scalability (increase alpha): method={method}, scoring={scoring}, n_entities={n_entities}, k={k}, alpha_values={alpha_values}")
    if method == "aqs":
        print(f"AQS initial package sample size: {args.aqs_package_limit}")
    print(f"Output: {out_file}")

    # For AQS: keep the same initial package set across all alpha values for comparability
    aqs_initial_packages: list[Package] | None = None
    if method == "aqs":
        aqs_initial_packages = sample_initial_packages(entity_ids, k, args.aqs_package_limit, seed=args.seed)

    rows: list[dict] = []
    for alpha in alpha_values:
        llm_evaluator = LLMEvaluator(mock_api=True, use_MGT=False)
        if method == "aqs":
            algorithm = AQSAlgorithm(
                entities=entities,
                components=components,
                k=k,
                alpha=float(alpha),
                query=query,
                llm_evaluator=llm_evaluator,
                initial_packages=aqs_initial_packages,
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
                "alpha": float(alpha),
                "time_maintain_packages": metadata.get("time_maintain_packages", 0),
                "time_ask_next_question": metadata.get("time_ask_next_question", 0),
                "time_process_response": metadata.get("time_process_response", 0),
                "time_total": metadata.get("time_total", 0),
            }
        )
        print(f"  alpha={alpha}: total={rows[-1]['time_total']}")

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "#entities",
                "k",
                "alpha",
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

