"""
Quality experiments — Section 2 — Cost (#probes).

Generates a CSV table:
  - rows: methods
  - columns: scoring functions f1..f6
  - cell value: #probes (number of NEW component evaluations / cache misses)

We treat:
  #probes == number of times LLMEvaluator.evaluate_component is called on a cache miss
  (works the same for MGT and real LLM calls; here we use MGT for speed/determinism).

K is configured at the top of the file. Output filename includes k.

Run from project root:
  python experiments/quality/section_2/run_quality_section_2_num_probes.py
"""

from __future__ import annotations

import csv
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path
from preprocessing.load_data import load_entities_from_csv
from utils.llm_interface import LLMEvaluator
from preprocessing.MGT import MGT
from PCS.algorithm import PCSAlgorithm
from AQS.algorithm import AQSAlgorithm, QuestionSelectionStrategy
from BASELINE.algorithm import ExactBaseline


# -----------------------------
# EXPERIMENT CONFIG (edit here)
# -----------------------------
N_ENTITIES = 8
K = 3  # change to 6 for the second table
ALPHA = 0.8

# MGT generation settings (per scoring, generated once per scoring if needed)
UNIFORM_EVERY_K = 10
MODEL = "gpt-4o-mini"
MOCK_API = False

MGT_DIR = ROOT / "mgt_Results"  # no subfolders
OUTPUT_DIR = ROOT / "experiments_outputs" / "quality_outputs" / "section_2"

# Methods (rows)
METHODS = ["PCS", "AQS", "Random", "GreedyLoose", "GreedyTight", "Exact"]

# For reproducibility of Random/Greedy variants
GLOBAL_SEED = 12345


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _mgt_slug(s: str) -> str:
    # Must match preprocessing/MGT.py::_slug exactly
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


def _entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def _load_entities_for_scoring(scoring: str, n_entities: int) -> Tuple[dict, str, list, Path]:
    cfg = get_config(scoring)
    query = cfg["query"]
    components = cfg["components"]
    dataset_path = get_dataset_path(scoring)

    entities_all = load_entities_from_csv(str(dataset_path))
    entity_ids = sorted(entities_all.keys(), key=_entity_order_key)[:n_entities]
    if len(entity_ids) < n_entities:
        raise ValueError(f"Dataset {dataset_path} has only {len(entity_ids)} entities; requested {n_entities}")
    entities = {eid: entities_all[eid] for eid in entity_ids}
    return entities, query, components, dataset_path


def _fresh_mgt_evaluator(dataset_path: Path, components: list, n_entities: int) -> LLMEvaluator:
    llm = LLMEvaluator(
        use_MGT=True,
        entities_csv_path=str(dataset_path),
        components=components,
        output_dir=str(MGT_DIR),
        n=n_entities,
    )
    llm.reset_component_evals_count()
    return llm


def _run_method(
    method: str,
    *,
    entities: dict,
    query: str,
    components: list,
    dataset_path: Path,
    n_entities: int,
    k: int,
    alpha: float,
) -> int:
    method = method.strip()

    # Fresh evaluator per method so caches don't leak across methods.
    llm = _fresh_mgt_evaluator(dataset_path, components, n_entities)

    if method == "PCS":
        algo = PCSAlgorithm(components=components, llm_evaluator=llm)
        algo.run(entities, k, query)
        return llm.get_component_evals_count()

    if method == "Exact":
        baseline = ExactBaseline(components, llm)
        baseline.run(entities, query, k)
        return llm.get_component_evals_count()

    # AQS-family
    if method == "AQS":
        strat = QuestionSelectionStrategy.HEURISTIC
    elif method == "Random":
        strat = QuestionSelectionStrategy.RANDOM
    elif method == "GreedyLoose":
        strat = QuestionSelectionStrategy.GREEDY_LOOSE
    elif method == "GreedyTight":
        strat = QuestionSelectionStrategy.GREEDY_TIGHT
    else:
        raise ValueError(f"Unknown method: {method}")

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
    )
    algo.run()
    return llm.get_component_evals_count()


def main() -> None:
    n_entities = int(N_ENTITIES)
    k = int(K)
    alpha = float(ALPHA)

    if k > n_entities:
        raise SystemExit("K cannot be greater than N_ENTITIES")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"quality_section_2_num_probes_k{k}_{ts}.csv"

    scorings = [f"f{i}" for i in range(1, 7)]
    _log(f"START section_2_num_probes: n_entities={n_entities} k={k} alpha={alpha:g} methods={METHODS}")

    # table[method][scoring] = probes
    table: Dict[str, Dict[str, int]] = {m: {} for m in METHODS}

    for scoring in scorings:
        random.seed(GLOBAL_SEED)  # stable per scoring
        _ensure_mgt_for_scoring(scoring, n_entities=n_entities, mgt_dir=MGT_DIR)
        entities, query, components, dataset_path = _load_entities_for_scoring(scoring, n_entities=n_entities)

        for method in METHODS:
            # Ensure deterministic randomness for Random/Greedy across methods
            random.seed(GLOBAL_SEED + hash((scoring, method)) % 10_000)
            probes = _run_method(
                method,
                entities=entities,
                query=query,
                components=components,
                dataset_path=dataset_path,
                n_entities=n_entities,
                k=k,
                alpha=alpha,
            )
            table[method][scoring] = int(probes)
            _log(f"[k={k}] {method} {scoring}: probes={probes}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", *scorings])
        for method in METHODS:
            w.writerow([method, *[table[method].get(s, 0) for s in scorings]])

    print(f"Wrote {out_path}")
    print(f"n_entities={n_entities} k={k} alpha={alpha:g}")


if __name__ == "__main__":
    main()

