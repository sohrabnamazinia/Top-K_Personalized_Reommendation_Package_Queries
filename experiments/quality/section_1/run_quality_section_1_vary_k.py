"""
Quality experiments — Section 1 — Vary k (PCS only).

Table format (to match the paper tables):
  - rows: PCS only
  - columns: f1..f6

For each scoring function f1..f6:
  1) Ensure MGT exists for that scoring (generated into mgt_Results/, no subfolders).
  2) Sweep k in K_VALUES. For each k:
      - Run PCS once to get a package.
      - Check if PCS package is alpha-top under exact (MGT) using:
            P(score(PCS_pkg) >= score(all other packages)) >= ALPHA
        (same probability logic as AQS PackageManager).
  3) Cell value = precision over k:
        precision = (#k where PCS is alpha-top) / (len(K_VALUES))

Output:
  experiments_outputs/quality_outputs/section_1/quality_section_1_vary_k_<timestamp>.csv

Run from project root:
  python experiments/quality/section_1/run_quality_section_1_vary_k.py
"""

from __future__ import annotations

import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path
from preprocessing.load_data import load_entities_from_csv
from utils.llm_interface import LLMEvaluator
from utils.models import Package
from preprocessing.MGT import MGT
from PCS.algorithm import PCSAlgorithm
from BASELINE.algorithm import ExactBaseline
from AQS.package_manager import PackageManager


# -----------------------------
# EXPERIMENT CONFIG (edit here)
# -----------------------------
N_ENTITIES = 10
K_VALUES = [3, 5, 7, 9]
ALPHA = 0.8

# MGT generation settings (per scoring, generated once per scoring if needed)
UNIFORM_EVERY_K = 10
MODEL = "gpt-4o-mini"
MOCK_API = False

# Where MGT files are written/read (NO subfolders)
MGT_DIR = ROOT / "mgt_Results"

# Where the output CSV table is written
OUTPUT_DIR = ROOT / "experiments_outputs" / "quality_outputs" / "section_1"

# Safety: exact probability check needs ALL packages. Keep this small.
MAX_PACKAGES_FOR_EXACT = 50_000


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
    cfg = get_config(scoring)
    for comp in cfg["components"]:
        slug = _mgt_slug(comp.description)
        expected = mgt_dir / f"MGT_{comp.name}_{slug}_{n_entities}.csv"
        if not expected.exists():
            return False
    return True


def _clean_mgt_dir_for_scoring(mgt_dir: Path, scoring: str, n_entities: int) -> None:
    """
    Component names repeat across scorings (c1, c2, ...). MGT loader matches by component name + n,
    so before generating for a scoring we remove any exact-n files for that scoring's component names.
    """
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

    _log(
        f"[{scoring}] Generating MGT (n={n_entities}, uniform_every_k={UNIFORM_EVERY_K}, mock_api={MOCK_API})"
    )
    t0 = time.perf_counter()
    mgt.generate(llm_for_generation, query, output_dir=str(mgt_dir), n=n_entities)
    _log(f"[{scoring}] MGT generation done in {time.perf_counter() - t0:.2f}s")


def entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def _alpha_top(pm: PackageManager, pkg: Package | None, alpha: float) -> bool:
    if pkg is None:
        return False
    return pm.check_alpha_top_package(pkg, alpha=alpha)


def _build_exact_pm(
    *,
    baseline: ExactBaseline,
    entities: dict,
    query: str,
    components: list,
    k: int,
) -> PackageManager:
    all_packages = baseline.build_packages(entities, k)
    if not all_packages:
        raise RuntimeError(f"No packages for k={k} with n_entities={len(entities)}")
    if len(all_packages) > MAX_PACKAGES_FOR_EXACT:
        raise SystemExit(
            f"Too many packages for exact probability check: C(n={len(entities)}, k={k})={len(all_packages)} > {MAX_PACKAGES_FOR_EXACT}.\n"
            f"Reduce N_ENTITIES or k, or increase MAX_PACKAGES_FOR_EXACT."
        )

    pm = PackageManager(entities=list(entities.keys()), k=k, components=components, packages=all_packages)
    for pkg in all_packages:
        lb, ub, _ = baseline.scoring.compute_package_score(pkg, entities, query, use_cache=True)
        pm.set_bounds(pkg, lb, ub)
    return pm


def _run_pcs(llm: LLMEvaluator, components: list, entities: dict, query: str, k: int) -> Package | None:
    algo = PCSAlgorithm(components=components, llm_evaluator=llm)
    pkg, _ = algo.run(entities, k, query)
    return pkg


def main() -> None:
    n_entities = int(N_ENTITIES)
    if n_entities <= 1:
        raise SystemExit("N_ENTITIES must be >= 2")
    if any(k <= 0 for k in K_VALUES):
        raise SystemExit("All K_VALUES must be positive")
    alpha = float(ALPHA)
    if alpha < 0.0 or alpha > 1.0:
        raise SystemExit("ALPHA must be in [0, 1]")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"quality_section_1_vary_k_{ts}.csv"

    scorings = [f"f{i}" for i in range(1, 7)]
    _log(
        f"START vary_k (PCS only): n_entities={n_entities} k_values={K_VALUES} alpha={alpha:g} "
        f"uniform_every_k={UNIFORM_EVERY_K} model={MODEL} mock_api={MOCK_API}"
    )

    # table[scoring] = precision over k
    table: Dict[str, float] = {}

    for scoring in scorings:
        _ensure_mgt_for_scoring(scoring, n_entities=n_entities, mgt_dir=MGT_DIR)

        cfg = get_config(scoring)
        query = cfg["query"]
        components = cfg["components"]
        dataset_path = get_dataset_path(scoring)

        entities_all = load_entities_from_csv(str(dataset_path))
        entity_ids = sorted(entities_all.keys(), key=entity_order_key)[:n_entities]
        if len(entity_ids) < n_entities:
            raise ValueError(f"Dataset {dataset_path} has only {len(entity_ids)} entities; requested {n_entities}")
        entities = {eid: entities_all[eid] for eid in entity_ids}

        llm = LLMEvaluator(
            use_MGT=True,
            entities_csv_path=str(dataset_path),
            components=components,
            output_dir=str(MGT_DIR),
            n=n_entities,
        )

        baseline = ExactBaseline(components, llm)
        # Probe once per scoring (covers any k).
        baseline._probe_all_component_values(entities, query)  # reuse cache for exact scoring

        ok_count_k = 0
        for k in K_VALUES:
            if k > n_entities:
                continue

            _log(f"[{scoring}] k={k}: building exact package bounds")
            pm = _build_exact_pm(baseline=baseline, entities=entities, query=query, components=components, k=k)

            _log(f"[{scoring}] k={k}: running PCS once")
            pcs_pkg = _run_pcs(llm, components, entities, query, k=k)

            ok = _alpha_top(pm, pcs_pkg, alpha=alpha)
            ok_count_k += 1 if ok else 0
            _log(f"[{scoring}] k={k}: PCS alpha-top={ok}")

        precision = ok_count_k / len(K_VALUES) if K_VALUES else 0.0
        table[scoring] = precision
        _log(f"[{scoring}] PCS precision_over_k={precision:.3f}")

    # Write CSV: single row (PCS), columns are f1..f6
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *scorings])
        writer.writerow(["PCS", *[f"{table.get(s, 0.0):.3f}" for s in scorings]])

    print(f"Wrote {out_path}")
    print(f"N_ENTITIES={n_entities}  K_VALUES={K_VALUES}  ALPHA={alpha:g}")


if __name__ == "__main__":
    main()

