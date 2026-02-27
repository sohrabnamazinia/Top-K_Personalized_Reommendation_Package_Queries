"""
Quality experiments — Section 1 — Vary LLM model (PCS only).

Table format (to match the paper tables):
  - rows: PCS only
  - columns: f1..f6

For each scoring function f1..f6, evaluate PCS twice:
  - once using MGT generated with MODEL_SMALL (e.g., gpt-4o-mini)
  - once using MGT generated with MODEL_LARGE (e.g., gpt-4o)

Correctness check (wrt exact under the SAME MGT):
  P(score(PCS_pkg) >= score(all other packages)) >= ALPHA
using the same probability logic as AQS PackageManager.

Cell value for each scoring f:
  precision = (#models where PCS is alpha-top) / 2

Implementation detail:
MGT filenames don't include model, so we cache each model's MGT files in a separate cache folder
and copy them into mgt_Results/ when running that model.

Run from project root:
  python experiments/quality/section_1/run_quality_section_1_vary_model.py
"""

from __future__ import annotations

import csv
import re
import shutil
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
K = 3
ALPHA = 0.8

MODEL_SMALL = "gpt-4o-mini"
MODEL_LARGE = "gpt-4o"
MODELS = [MODEL_SMALL, MODEL_LARGE]

# When generating MGT: do a real LLM call every k rows; others are synthetic
UNIFORM_EVERY_K = 10
MOCK_API = False

# Working MGT dir used by LLMEvaluator(use_MGT=True) (NO subfolders)
MGT_DIR = ROOT / "mgt_Results"

# Cache dir to store per-model MGTs safely (not used by LLMEvaluator directly)
CACHE_DIR = ROOT / "mgt_cache_quality_section_1_vary_model"

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


def _expected_mgt_paths(scoring: str, n_entities: int, base_dir: Path) -> List[Path]:
    cfg = get_config(scoring)
    out = []
    for comp in cfg["components"]:
        slug = _mgt_slug(comp.description)
        out.append(base_dir / f"MGT_{comp.name}_{slug}_{n_entities}.csv")
    return out


def _has_mgt(scoring: str, n_entities: int, base_dir: Path) -> bool:
    return all(p.exists() for p in _expected_mgt_paths(scoring, n_entities, base_dir))


def _clean_working_mgt(scoring: str, n_entities: int) -> None:
    """
    Remove exact-n MGT files for this scoring's component names from MGT_DIR.
    Needed because component names repeat across scorings and models.
    """
    cfg = get_config(scoring)
    for comp in cfg["components"]:
        for p in MGT_DIR.glob(f"MGT_{comp.name}_*_{n_entities}.csv"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def _cache_subdir(scoring: str, model: str, n_entities: int) -> Path:
    safe_model = re.sub(r"[^\w.-]+", "_", model)
    return CACHE_DIR / f"n{n_entities}" / scoring / safe_model


def _stage_cached_mgt_into_workdir(scoring: str, model: str, n_entities: int) -> None:
    """
    Ensure MGT_DIR contains the right MGT files for (scoring, model, n_entities),
    either by copying from cache or generating and then caching.
    """
    cache_dir = _cache_subdir(scoring, model, n_entities)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If cache exists, just copy into workdir.
    if _has_mgt(scoring, n_entities, cache_dir):
        _log(f"[{scoring}] Using cached MGT for model={model} from {cache_dir}")
        _clean_working_mgt(scoring, n_entities)
        MGT_DIR.mkdir(parents=True, exist_ok=True)
        for src in _expected_mgt_paths(scoring, n_entities, cache_dir):
            shutil.copy2(src, MGT_DIR / src.name)
        return

    # Otherwise generate into workdir, then copy into cache.
    _log(f"[{scoring}] Generating MGT for model={model} (n={n_entities})")
    _clean_working_mgt(scoring, n_entities)
    MGT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = get_config(scoring)
    query = cfg["query"]
    components = cfg["components"]
    dataset_path = get_dataset_path(scoring)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    images_base_path = None
    if scoring in ("f5", "f6"):
        images_base_path = str(ROOT / "data" / "yelp_dataset" / "photos")

    llm_for_generation = LLMEvaluator(
        mock_api=bool(MOCK_API),
        use_MGT=False,
        model=str(model),
        images_base_path=images_base_path,
    )
    mgt = MGT(
        entities_csv_path=str(dataset_path),
        components=components,
        uniform_every_k=int(UNIFORM_EVERY_K),
    )

    t0 = time.perf_counter()
    mgt.generate(llm_for_generation, query, output_dir=str(MGT_DIR), n=n_entities)
    _log(f"[{scoring}] MGT generated in {time.perf_counter() - t0:.2f}s; caching to {cache_dir}")

    # Verify generation
    if not _has_mgt(scoring, n_entities, MGT_DIR):
        raise FileNotFoundError(f"[{scoring}] Expected MGT files not found after generation in {MGT_DIR}")

    for src in _expected_mgt_paths(scoring, n_entities, MGT_DIR):
        shutil.copy2(src, cache_dir / src.name)


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
    k = int(K)
    alpha = float(ALPHA)
    if k > n_entities:
        raise SystemExit("K cannot be greater than N_ENTITIES")
    if alpha < 0.0 or alpha > 1.0:
        raise SystemExit("ALPHA must be in [0, 1]")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"quality_section_1_vary_model_{ts}.csv"

    scorings = [f"f{i}" for i in range(1, 7)]
    _log(
        f"START vary_model (PCS only): n_entities={n_entities} k={k} alpha={alpha:g} models={MODELS} "
        f"uniform_every_k={UNIFORM_EVERY_K} mock_api={MOCK_API}"
    )

    table: Dict[str, float] = {}

    for scoring in scorings:
        ok_models = 0

        for model in MODELS:
            _stage_cached_mgt_into_workdir(scoring, model=model, n_entities=n_entities)

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
            baseline._probe_all_component_values(entities, query)

            pm = _build_exact_pm(baseline=baseline, entities=entities, query=query, components=components, k=k)
            pcs_pkg = _run_pcs(llm, components, entities, query, k=k)

            ok = _alpha_top(pm, pcs_pkg, alpha=alpha)
            ok_models += 1 if ok else 0
            _log(f"[{scoring}] model={model}: PCS alpha-top={ok}")

        precision = ok_models / len(MODELS) if MODELS else 0.0
        table[scoring] = precision
        _log(f"[{scoring}] PCS precision_over_models={precision:.3f}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *scorings])
        writer.writerow(["PCS", *[f"{table.get(s, 0.0):.3f}" for s in scorings]])

    print(f"Wrote {out_path}")
    print(f"N_ENTITIES={n_entities}  K={k}  ALPHA={alpha:g}  MODELS={MODELS}")


if __name__ == "__main__":
    main()

