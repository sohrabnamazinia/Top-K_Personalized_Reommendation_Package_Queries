"""
Quality experiments — Section 1 — Vary budget.

We evaluate budget-limited variants of algorithms using MGT (no live LLM needed after MGT is generated).

Budget definition:
  budget = maximum number of NEW component evaluations (cache misses) allowed during an algorithm run.
  When budget is exhausted, the algorithm stops immediately and returns its current best package.

Algorithms:
  - PCS: budget-limited; returns current package when budget exhausted
  - AQS (heuristic), Random, GreedyLoose, GreedyTight: budget-limited; returns current super-candidate
  - Exact: N/A (not budget limited)

Correctness check for returned package (wrt exact under same MGT):
  P(score(pkg) >= score(all other packages)) >= ALPHA
using AQS PackageManager probability logic with exact bounds (computed via ExactBaseline scoring).

Output:
  experiments_outputs/quality_outputs/section_1/quality_section_1_vary_budget_<timestamp>.csv
  Rows: PCS, AQS, Random, GreedyLoose, GreedyTight, Exact
  Columns: f1..f6
  Cell: precision over BUDGET_VALUES (ok_count / len(BUDGET_VALUES)), except Exact = "N/A"

Run from project root:
  python experiments/quality/section_1/run_quality_section_1_vary_budget.py
"""

from __future__ import annotations

import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path
from preprocessing.load_data import load_entities_from_csv
from utils.llm_interface import LLMEvaluator
from preprocessing.MGT import MGT
from utils.models import Package
from PCS.algorithm import PCSAlgorithm
from AQS.algorithm import AQSAlgorithm, QuestionSelectionStrategy
from AQS.package_manager import PackageManager
from BASELINE.algorithm import ExactBaseline


# -----------------------------
# EXPERIMENT CONFIG (edit here)
# -----------------------------
N_ENTITIES = 10
K = 3
ALPHA = 0.5
BUDGET_VALUES = [10, 50, 100, 200, 500]

# MGT generation settings (per scoring, generated once per scoring if needed)
UNIFORM_EVERY_K = 100
MODEL = "gpt-4o-mini"
MOCK_API = False

MGT_DIR = ROOT / "mgt_Results"
OUTPUT_DIR = ROOT / "experiments_outputs" / "quality_outputs" / "section_1"

MAX_PACKAGES_FOR_EXACT = 5_000

METHODS = ["PCS", "AQS", "Random", "GreedyLoose", "GreedyTight", "Exact"]


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _mgt_slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s[:64] if len(s) > 64 else s


def _has_mgt_for_scoring(scoring: str, n_entities: int, mgt_dir: Path) -> bool:
    cfg = get_config(scoring)
    for comp in cfg["components"]:
        slug = _mgt_slug(comp.description)
        expected = mgt_dir / f"MGT_{comp.name}_{slug}_{n_entities}.csv"
        if not expected.exists():
            return False
    return True


def _clean_mgt_dir_for_scoring(mgt_dir: Path, scoring: str, n_entities: int) -> None:
    cfg = get_config(scoring)
    for comp in cfg["components"]:
        for p in mgt_dir.glob(f"MGT_{comp.name}_*_{n_entities}.csv"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def _ensure_mgt_for_scoring(scoring: str, n_entities: int, mgt_dir: Path) -> None:
    scoring = scoring.lower()
    cfg = get_config(scoring)
    query = cfg["query"]
    components = cfg["components"]
    dataset_path = get_dataset_path(scoring)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    mgt_dir.mkdir(parents=True, exist_ok=True)
    if _has_mgt_for_scoring(scoring, n_entities, mgt_dir):
        _log(f"[{scoring}] MGT exists for n={n_entities}. Skipping generation.")
        return

    _clean_mgt_dir_for_scoring(mgt_dir, scoring=scoring, n_entities=n_entities)

    images_base_path = None
    if scoring in ("f5", "f6"):
        images_base_path = str(ROOT / "data" / "yelp_dataset" / "photos")

    llm_for_generation = LLMEvaluator(
        mock_api=bool(MOCK_API),
        use_MGT=False,
        model=str(MODEL),
        images_base_path=images_base_path,
    )
    mgt = MGT(
        entities_csv_path=str(dataset_path),
        components=components,
        uniform_every_k=int(UNIFORM_EVERY_K),
    )

    _log(f"[{scoring}] Generating MGT (n={n_entities}, uniform_every_k={UNIFORM_EVERY_K}, mock_api={MOCK_API})")
    t0 = time.perf_counter()
    mgt.generate(llm_for_generation, query, output_dir=str(mgt_dir), n=n_entities)
    _log(f"[{scoring}] MGT generation done in {time.perf_counter() - t0:.2f}s")


def entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def _build_exact_pm(baseline: ExactBaseline, entities: dict, query: str, components: list, k: int) -> PackageManager:
    all_packages = baseline.build_packages(entities, k)
    if not all_packages:
        raise RuntimeError("No packages built")
    if len(all_packages) > MAX_PACKAGES_FOR_EXACT:
        raise SystemExit(
            f"Too many packages for exact check: C(n={len(entities)}, k={k})={len(all_packages)} > {MAX_PACKAGES_FOR_EXACT}.\n"
            f"Reduce N_ENTITIES or k."
        )
    pm = PackageManager(entities=list(entities.keys()), k=k, components=components, packages=all_packages)
    for pkg in all_packages:
        lb, ub, _ = baseline.scoring.compute_package_score(pkg, entities, query, use_cache=True)
        pm.set_bounds(pkg, lb, ub)
    return pm


def _is_alpha_top(pm: PackageManager, pkg: Optional[Package], alpha: float) -> bool:
    if pkg is None:
        return False
    return pm.check_alpha_top_package(pkg, alpha=alpha)


def _run_budgeted_method(
    method: str,
    *,
    entities: dict,
    query: str,
    components: list,
    llm: LLMEvaluator,
    k: int,
    alpha: float,
    budget: int,
) -> Optional[Package]:
    if method == "PCS":
        algo = PCSAlgorithm(components=components, llm_evaluator=llm, enable_budget_limit=True, budget=budget)
        pkg, _ = algo.run(entities, k, query)
        return pkg
    if method in ("AQS", "Random", "GreedyLoose", "GreedyTight"):
        strat = QuestionSelectionStrategy.HEURISTIC
        if method == "Random":
            strat = QuestionSelectionStrategy.RANDOM
        elif method == "GreedyLoose":
            strat = QuestionSelectionStrategy.GREEDY_LOOSE
        elif method == "GreedyTight":
            strat = QuestionSelectionStrategy.GREEDY_TIGHT
        algo = AQSAlgorithm(
            entities=entities,
            components=components,
            k=k,
            alpha=alpha,
            query=query,
            llm_evaluator=llm,
            initial_packages=None,
            print_log=False,
            init_dim_1=True,
            selection_strategy=strat,
            enable_budget_limit=True,
            budget=budget,
        )
        pkg, _ = algo.run()
        return pkg
    if method == "Exact":
        return None
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    n_entities = int(N_ENTITIES)
    k = int(K)
    alpha = float(ALPHA)
    if k > n_entities:
        raise SystemExit("K cannot be greater than N_ENTITIES")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"quality_section_1_vary_budget_{ts}.csv"

    scorings = [f"f{i}" for i in range(1, 7)]
    _log(f"START vary_budget: n_entities={n_entities} k={k} alpha={alpha:g} budgets={BUDGET_VALUES}")

    table: Dict[str, Dict[str, str]] = {m: {} for m in METHODS}

    for scoring in scorings:
        _ensure_mgt_for_scoring(scoring, n_entities=n_entities, mgt_dir=MGT_DIR)

        cfg = get_config(scoring)
        query = cfg["query"]
        components = cfg["components"]
        dataset_path = get_dataset_path(scoring)

        entities_all = load_entities_from_csv(str(dataset_path))
        entity_ids = sorted(entities_all.keys(), key=entity_order_key)[:n_entities]
        entities = {eid: entities_all[eid] for eid in entity_ids}

        # Exact baseline evaluator (no budget) for probability bounds
        llm_exact = LLMEvaluator(
            use_MGT=True,
            entities_csv_path=str(dataset_path),
            components=components,
            output_dir=str(MGT_DIR),
            n=n_entities,
        )
        baseline = ExactBaseline(components, llm_exact)
        baseline._probe_all_component_values(entities, query)

        pm = _build_exact_pm(baseline, entities, query, components, k=k)

        for method in METHODS:
            if method == "Exact":
                table[method][scoring] = "N/A"
                continue

            ok = 0
            for b in BUDGET_VALUES:
                # Fresh evaluator per run so caches don't leak across budgets
                llm = LLMEvaluator(
                    use_MGT=True,
                    entities_csv_path=str(dataset_path),
                    components=components,
                    output_dir=str(MGT_DIR),
                    n=n_entities,
                )
                pkg = _run_budgeted_method(
                    method,
                    entities=entities,
                    query=query,
                    components=components,
                    llm=llm,
                    k=k,
                    alpha=alpha,
                    budget=int(b),
                )
                if _is_alpha_top(pm, pkg, alpha=alpha):
                    ok += 1
            precision = ok / len(BUDGET_VALUES) if BUDGET_VALUES else 0.0
            table[method][scoring] = f"{precision:.3f}"
            _log(f"[{scoring}] {method} precision_over_budget={precision:.3f}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *scorings])
        for method in METHODS:
            writer.writerow([method, *[table[method].get(s, "") for s in scorings]])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


