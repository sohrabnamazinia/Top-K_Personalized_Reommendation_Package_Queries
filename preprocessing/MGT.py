"""
Materialized Ground Truth (MGT): materialize LLM component values to CSV for reuse without LLM calls.
Generate is a one-time process; when using MGT we only read from the generated CSVs.
"""
import csv
import glob
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

from utils.models import Component, Entity
from utils.llm_interface import LLMEvaluator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.load_data import load_entities_from_csv


def _slug(s: str) -> str:
    """Sanitize string for use in filename (alphanumeric and underscores)."""
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[-\s]+', '_', s).strip('_')
    return s[:64] if len(s) > 64 else s


def _synthetic_time() -> float:
    """Synthetic time in seconds: mostly 0.4--1.0, with low prob a bit lower or higher."""
    r = random.random()
    if r < 0.05:
        return round(random.uniform(0.25, 0.4), 4)
    if r > 0.95:
        return round(random.uniform(1.0, 1.2), 4)
    return round(random.uniform(0.4, 1.0), 4)


def _synthetic_score() -> Tuple[float, float]:
    """Synthetic score: lb = ub, random in [0.0, 1.0]."""
    val = round(random.uniform(0.0, 1.0), 1)
    return (val, val)


def _entity_id_sort_key(eid: str) -> int:
    """Sort key for entity ids: e1, e2, ..., e10, e11 (numeric order)."""
    m = re.search(r"(\d+)$", eid)
    return int(m.group(1)) if m else 0


def _entity_number(eid: str) -> int:
    """Extract numeric part from entity id (e.g. e5 -> 5). Assumes e1..en."""
    m = re.search(r"(\d+)$", eid)
    return int(m.group(1)) if m else 0


def _row_index_dim1(eid: str) -> int:
    """Row index for dim 1: e1->0, e2->1, etc. Assumes entities are e1..en."""
    return _entity_number(eid) - 1


def _row_index_dim2(eid1: str, eid2: str, n: int) -> int:
    """Row index for dim 2 (upper-triangle pairs). i,j are 1-based entity numbers."""
    i, j = _entity_number(eid1), _entity_number(eid2)
    if i > j:
        i, j = j, i
    return (i - 1) * n - i * (i - 1) // 2 + (j - i - 1)


class MGT:
    """
    Materialized Ground Truth: generate CSVs (one-time) or read from them.
    When only reading, we only use the generated CSV files; no entities CSV needed for fetch.
    """

    def __init__(
        self,
        entities_csv_path: str,
        components: List[Component],
        uniform_every_k: int = 1,
    ):
        """
        Args:
            entities_csv_path: Path to entities CSV (used only by generate()).
            components: List of Component (name, description, dimension 1 or 2).
            uniform_every_k: Do real LLM call only for 1 out of every k rows (generate only).
        """
        self.entities_csv_path = entities_csv_path
        self.components = list(components)
        self.uniform_every_k = max(1, int(uniform_every_k))
        self._csv_paths: Dict[str, str] = {}
        self._n: int = 0  # number of entities (parsed from filename)

    def _n_from_path(self, path: str) -> int:
        m = re.search(r"_(\d+)\.csv$", path)
        return int(m.group(1)) if m else 0

    def ensure_mgt_for_n(self, output_dir: str, n: int) -> None:
        """
        Ensure MGT CSVs for exactly n entities exist. If not, build them by slicing
        a larger-n MGT file (same component name/desc) if one exists.
        """
        os.makedirs(output_dir, exist_ok=True)
        for comp in self.components:
            pattern = os.path.join(output_dir, f"MGT_{comp.name}_*.csv")
            candidates = glob.glob(pattern)
            # Check if we already have exact n
            for path in candidates:
                if self._n_from_path(path) == n:
                    break
            else:
                # No exact match: find smallest file with n_large >= n
                with_n = [(p, self._n_from_path(p)) for p in candidates if self._n_from_path(p) >= n]
                if not with_n:
                    raise FileNotFoundError(
                        f"No MGT CSV for component '{comp.name}' with n>={n}. "
                        f"Found only: {[self._n_from_path(p) for p in candidates]}. Run generate() for at least n={n} entities first."
                    )
                source_path = min(with_n, key=lambda x: x[1])[0]
                slug_match = re.search(
                    r"MGT_" + re.escape(comp.name) + r"_(.+)_\d+\.csv$",
                    os.path.basename(source_path),
                )
                slug = slug_match.group(1) if slug_match else _slug(comp.description)
                out_fname = f"MGT_{comp.name}_{slug}_{n}.csv"
                out_path = os.path.join(output_dir, out_fname)
                with open(source_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    rows = list(reader)
                if comp.dimension == 1:
                    take = min(n, len(rows))
                    subset = rows[:take]
                else:
                    take = min(n * (n - 1) // 2, len(rows))
                    subset = rows[:take]
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(subset)

    def load_from_existing(self, output_dir: str = "mgt_Results", n: Optional[int] = None) -> None:
        """
        Find MGT CSVs for each component for entity count n.
        n: required when multiple MGT files may exist; only files ending _n.csv are used.
        If n is None, uses the file with largest n per component (backward compat).
        Raises FileNotFoundError if a matching CSV is missing.
        """
        self._csv_paths = {}
        missing = []
        for comp in self.components:
            pattern = os.path.join(output_dir, f"MGT_{comp.name}_*.csv")
            candidates = glob.glob(pattern)
            if not candidates:
                missing.append(pattern if n is None else f"MGT for n={n} not found: {pattern}")
                continue
            if n is not None:
                matching = [p for p in candidates if self._n_from_path(p) == n]
                if not matching:
                    missing.append(f"MGT for n={n} not found: {pattern}")
                    continue
                path = matching[0]
                self._n = n
            else:
                path = max(candidates, key=self._n_from_path)
                n_parsed = self._n_from_path(path)
                if n_parsed:
                    self._n = n_parsed
            self._csv_paths[comp.name] = path
        if missing:
            raise FileNotFoundError(
                "use_MGT is set to True but MGT CSV files do not exist: " + "; ".join(missing)
            )

    def generate(
        self,
        llm_evaluator: LLMEvaluator,
        query: str,
        output_dir: str = "mgt_Results",
        n: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        One-time process: run LLM calls (or 1/uniform_every_k), write one CSV per component.
        Uses entities_csv_path only here to load entities and build row order.

        Args:
            n: If set, use only the first n entities (after sorting by entity_id). If None, use all entities in the CSV.
        """
        entities_all = load_entities_from_csv(self.entities_csv_path)
        entity_ids_sorted = sorted(entities_all.keys(), key=_entity_id_sort_key)
        if n is not None:
            entity_ids_sorted = entity_ids_sorted[:n]
        entities = {eid: entities_all[eid] for eid in entity_ids_sorted}

        pairs_sorted = []
        for i in range(len(entity_ids_sorted)):
            for j in range(i + 1, len(entity_ids_sorted)):
                pairs_sorted.append((entity_ids_sorted[i], entity_ids_sorted[j]))

        n = len(entities)
        self._n = n
        os.makedirs(output_dir, exist_ok=True)
        self._csv_paths = {}

        for comp in self.components:
            slug_desc = _slug(comp.description)
            fname = f"MGT_{comp.name}_{slug_desc}_{n}.csv"
            path = os.path.join(output_dir, fname)
            self._csv_paths[comp.name] = path

            if comp.dimension == 1:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["entity_id", "score_lb", "score_ub", "time"])
                    for idx, eid in enumerate(entity_ids_sorted):
                        do_call = (idx % self.uniform_every_k) == 0
                        if do_call:
                            cache_key = llm_evaluator._get_cache_key(comp, [eid], query)
                            lb, ub, elapsed = llm_evaluator._evaluate_component_llm(
                                comp, entities, [eid], query, use_cache=False, cache_key=cache_key
                            )
                            w.writerow([eid, lb, ub, f"{elapsed:.4f}"])
                        else:
                            lb, ub = _synthetic_score()
                            w.writerow([eid, lb, ub, f"{_synthetic_time():.4f}"])

            elif comp.dimension == 2:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["entity_id_1", "entity_id_2", "score_lb", "score_ub", "time"])
                    for idx, (e1, e2) in enumerate(pairs_sorted):
                        do_call = (idx % self.uniform_every_k) == 0
                        if do_call:
                            cache_key = llm_evaluator._get_cache_key(comp, [e1, e2], query)
                            lb, ub, elapsed = llm_evaluator._evaluate_component_llm(
                                comp, entities, [e1, e2], query, use_cache=False, cache_key=cache_key
                            )
                            w.writerow([e1, e2, lb, ub, f"{elapsed:.4f}"])
                        else:
                            lb, ub = _synthetic_score()
                            w.writerow([e1, e2, lb, ub, f"{_synthetic_time():.4f}"])

            else:
                raise ValueError(f"Unsupported component dimension: {comp.dimension}")

        return self._csv_paths.copy()

    def fetch(
        self,
        component_name: str,
        entity_ids: List[str],
    ) -> Tuple[float, float, float]:
        """
        Read (score_lb, score_ub, time) from the component's MGT CSV.
        Row index is computed via O(1) formula (assumes entities e1..en).
        """
        comp = next((c for c in self.components if c.name == component_name), None)
        if comp is None:
            raise ValueError(f"Unknown component_name: {component_name}")

        path = self._csv_paths.get(component_name)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"No MGT CSV for component '{component_name}'. Run generate() or load_from_existing() first."
            )

        if not self._n:
            raise FileNotFoundError("MGT not loaded. Call load_from_existing() first.")

        if comp.dimension == 1:
            if len(entity_ids) != 1:
                raise ValueError(
                    f"Component '{component_name}' (dim 1) requires exactly one entity_id"
                )
            row_idx = _row_index_dim1(entity_ids[0])
        else:
            if len(entity_ids) != 2:
                raise ValueError(
                    f"Component '{component_name}' (dim 2) requires exactly two entity_ids"
                )
            row_idx = _row_index_dim2(entity_ids[0], entity_ids[1], self._n)

        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        row = rows[row_idx]
        return (float(row["score_lb"]), float(row["score_ub"]), float(row["time"]))
