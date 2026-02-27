"""
Quality experiments — Section 1 — Vary alpha.

Creates a table (CSV) where:
  - rows are methods: PCS, AQS, Exact, Random, GreedyLoose, GreedyTight
  - columns are scoring functions: f1..f6

Each cell is PRECISION over a list of alpha values:
  precision = (# of runs where returned package is alpha-top) / (# alpha values)

We use MGT values (no real LLM, no mock):
  LLMEvaluator(use_MGT=True, ...)

By default we treat "alpha-top" under exact (MGT) scores as:
  returned package score == max score among all packages (ties allowed).
With exact MGT intervals, this matches the alpha-top definition used in PackageManager.

Run from project root:
  python experiments/quality/section_1/run_quality_section_1_vary_alpha.py
"""

from __future__ import annotations

import csv
import sys
import time
import re
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import CONFIGS, get_config, get_dataset_path
from preprocessing.load_data import load_entities_from_csv
from utils.llm_interface import LLMEvaluator
from utils.models import Package
from AQS.package_manager import PackageManager
from PCS.algorithm import PCSAlgorithm
from BASELINE.algorithm import ExactBaseline
from preprocessing.MGT import MGT


METHOD = "PCS"

# -----------------------------
# EXPERIMENT CONFIG (edit here)
# -----------------------------
N_ENTITIES = 20
K = 3
ALPHA_VALUES = [0.001, 0.01, 0.1, 0.9]

# MGT generation settings (per scoring, generated right before running that scoring)
UNIFORM_EVERY_K = 100  # do a real LLM call every k rows; others are synthetic
MODEL = "gpt-4o-mini"
MOCK_API = False  # True = random values, no external calls

# Where MGT files are written/read (NO subfolders)
MGT_DIR = ROOT / "mgt_Results"

# Where the output CSV table is written
OUTPUT_DIR = ROOT / "experiments_outputs" / "quality_outputs" / "section_1"


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _mgt_slug(s: str) -> str:
    """
    Must match preprocessing/MGT.py::_slug exactly, since filenames embed this slug.
    """
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s[:64] if len(s) > 64 else s


def _has_mgt_for_scoring(scoring: str, n_entities: int, mgt_dir: Path) -> bool:
    """
    True iff ALL expected exact-n MGT files for this scoring already exist in mgt_dir.
    We match on the exact filename pattern used by MGT.generate():
      MGT_{component_name}_{slug(description)}_{n}.csv
    """
    cfg = get_config(scoring)
    for comp in cfg["components"]:
        slug = _mgt_slug(comp.description)
        expected = mgt_dir / f"MGT_{comp.name}_{slug}_{n_entities}.csv"
        if not expected.exists():
            return False
    return True


def _clean_mgt_dir_for_scoring(mgt_dir: Path, scoring: str, n_entities: int) -> None:
    """
    IMPORTANT: component names repeat across scorings (c1, c2, ...). Since MGT loader matches by component name
    and n only, we must ensure there is only ONE exact-n file per component name in mgt_dir for the current scoring.
    """
    cfg = get_config(scoring)
    for comp in cfg["components"]:
        for p in mgt_dir.glob(f"MGT_{comp.name}_*_{n_entities}.csv"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def _generate_mgt_for_scoring(scoring: str, n_entities: int, mgt_dir: Path) -> None:
    """
    Generate MGT for this scoring into mgt_dir unless it already exists.
    """
    scoring = scoring.lower()
    cfg = get_config(scoring)
    query = cfg["query"]
    components = cfg["components"]
    dataset_path = get_dataset_path(scoring)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    mgt_dir.mkdir(parents=True, exist_ok=True)

    if _has_mgt_for_scoring(scoring, n_entities, mgt_dir):
        _log(f"[{scoring}] MGT already exists for n={n_entities} in {mgt_dir}. Skipping generation.")
        return

    _clean_mgt_dir_for_scoring(mgt_dir, scoring=scoring, n_entities=n_entities)

    # Yelp can optionally use images during real calls.
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

    _log(
        f"[{scoring}] Generating MGT into {mgt_dir} "
        f"(n={n_entities}, uniform_every_k={UNIFORM_EVERY_K}, mock_api={MOCK_API})"
    )
    t0 = time.perf_counter()
    mgt.generate(llm_for_generation, query, output_dir=str(mgt_dir), n=n_entities)
    _log(f"[{scoring}] MGT generation done in {time.perf_counter() - t0:.2f}s")


def entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def score_midpoint(baseline: ExactBaseline, pkg: Package, entities: dict, query: str) -> float:
    lb, ub, _ = baseline.scoring.compute_package_score(pkg, entities, query, use_cache=True)
    return (lb + ub) / 2.0


@dataclass
class ScoringContext:
    scoring: str
    entities: dict
    query: str
    components: list
    llm: LLMEvaluator
    baseline: ExactBaseline
    pm: PackageManager  # bounds set to "exact baseline" package-score intervals
    best_score: float
    best_packages: List[Package]


def build_context(scoring: str, n_entities: int, k: int, mgt_base_dir: Path) -> ScoringContext:
    scoring = scoring.lower()
    cfg = get_config(scoring)
    query = cfg["query"]
    components = cfg["components"]
    dataset_path = get_dataset_path(scoring)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    entities_all = load_entities_from_csv(str(dataset_path))
    entity_ids = sorted(entities_all.keys(), key=entity_order_key)[:n_entities]
    if len(entity_ids) < n_entities:
        raise ValueError(f"Dataset {dataset_path} has only {len(entity_ids)} entities; requested {n_entities}")
    entities = {eid: entities_all[eid] for eid in entity_ids}

    # IMPORTANT: using single shared mgt_Results directory (NO subfolders).
    mgt_dir = mgt_base_dir
    llm = LLMEvaluator(
        use_MGT=True,
        entities_csv_path=str(dataset_path),
        components=components,
        output_dir=str(mgt_dir),
        n=n_entities,
    )

    baseline = ExactBaseline(components, llm)

    # Exact baseline run gives us the max score (midpoint) among all packages of size k.
    best_pkg, scores_desc = baseline.run(entities, query, k)
    if not scores_desc:
        raise RuntimeError("No packages to score (check n_entities and k).")
    best_score = scores_desc[0]

    # Build ALL packages and compute their (lb, ub) under the exact baseline scoring.
    # We'll use these bounds to compute:
    #   P(score(pkg_returned) >= score(all other pkgs)) >= alpha
    all_packages = baseline.build_packages(entities, k)
    pm = PackageManager(entities=list(entities.keys()), k=k, components=components, packages=all_packages)

    # Collect all max-midpoint packages for tie-robustness.
    best_packages: List[Package] = []
    for pkg in all_packages:
        lb, ub, _ = baseline.scoring.compute_package_score(pkg, entities, query, use_cache=True)
        pm.set_bounds(pkg, lb, ub)
        s_mid = (lb + ub) / 2.0
        if abs(s_mid - best_score) <= 1e-9:
            best_packages.append(pkg)

    return ScoringContext(
        scoring=scoring,
        entities=entities,
        query=query,
        components=components,
        llm=llm,
        baseline=baseline,
        pm=pm,
        best_score=best_score,
        best_packages=best_packages,
    )


def is_alpha_top_exact(ctx: ScoringContext, pkg: Package, alpha: float) -> bool:
    if pkg is None:
        return False
    return ctx.pm.check_alpha_top_package(pkg, alpha=alpha)


def run_pcs(ctx: ScoringContext, k: int) -> Package | None:
    algo = PCSAlgorithm(components=ctx.components, llm_evaluator=ctx.llm)
    pkg, _ = algo.run(ctx.entities, k, ctx.query)
    return pkg


def main() -> None:
    alpha_values = list(ALPHA_VALUES)
    if not alpha_values:
        raise SystemExit("ALPHA_VALUES is empty")

    n_entities = int(N_ENTITIES)
    k = int(K)
    if k > n_entities:
        raise SystemExit("K cannot be greater than N_ENTITIES")

    mgt_dir = Path(MGT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"quality_section_1_vary_alpha_{ts}.csv"

    scorings = [f"f{i}" for i in range(1, 7)]

    table: Dict[str, float] = {}
    _log(
        f"START vary_alpha: n_entities={n_entities} k={k} alphas={alpha_values} "
        f"uniform_every_k={UNIFORM_EVERY_K} model={MODEL} mock_api={MOCK_API} mgt_dir={mgt_dir}"
    )

    for scoring in scorings:
        _generate_mgt_for_scoring(scoring, n_entities=n_entities, mgt_dir=mgt_dir)
        _log(f"[{scoring}] Building context (loads MGT + runs ExactBaseline)")
        ctx = build_context(scoring, n_entities=n_entities, k=k, mgt_base_dir=mgt_dir)

        _log(f"[{scoring}] Running PCS once")
        pkg_pcs = run_pcs(ctx, k=k)
        ok_count = 0
        for a in alpha_values:
            if is_alpha_top_exact(ctx, pkg_pcs, alpha=a):
                ok_count += 1
        precision = ok_count / len(alpha_values)
        table[scoring] = precision
        _log(f"[{scoring}] PCS precision={precision:.3f}")

    # Write CSV (rows = methods, columns = f1..f6)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *scorings])
        writer.writerow([METHOD, *[f"{table.get(s, 0.0):.3f}" for s in scorings]])

    print(f"Wrote {out_path}")
    print(f"alpha_values={alpha_values}  n_entities={n_entities}  k={k}")


if __name__ == "__main__":
    main()

