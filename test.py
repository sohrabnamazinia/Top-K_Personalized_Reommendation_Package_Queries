"""
Health check for data/yelp_dataset.csv.
1. Deduplicate by entity_name: keep one row per name (prefer non-empty image_id; else keep first).
2. Re-assign entity_id to e1, e2, ... and write back to CSV.
3. Run health check and print stats.
Run from project root: python test.py
"""
import csv
from pathlib import Path

DATA_CSV = Path(__file__).parent / "data" / "yelp_dataset.csv"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
CHECK = "\u2705"
CROSS = "\u274C"


def ok(msg: str) -> None:
    print(f"  {GREEN}{CHECK} {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"  {RED}{CROSS} {msg}{RESET}")


def run_health_check(rows: list[dict]) -> None:
    """Print health check for a list of row dicts (with keys entity_id, entity_name, data, image_id)."""
    entity_ids = [(r.get("entity_id") or "").strip() for r in rows]
    entity_names = [(r.get("entity_name") or "").strip() for r in rows]
    data_values = [(r.get("data") or "").strip() for r in rows]
    image_ids = [(r.get("image_id") or "").strip() for r in rows if (r.get("image_id") or "").strip()]

    n = len(rows)

    empty_eid = sum(1 for x in entity_ids if not x)
    unique_eid = len(set(entity_ids)) == n
    if empty_eid == 0 and unique_eid:
        ok(f"entity_id: all non-empty and unique ({n} rows)")
    else:
        if empty_eid:
            fail(f"entity_id: {empty_eid} empty")
        if not unique_eid:
            fail(f"entity_id: not unique ({len(set(entity_ids))} unique for {n} rows)")

    empty_name = sum(1 for x in entity_names if not x)
    unique_name = len(set(entity_names)) == n
    if empty_name == 0 and unique_name:
        ok(f"entity_name: all non-empty and unique ({n} rows)")
    else:
        if empty_name:
            fail(f"entity_name: {empty_name} empty")
        if not unique_name:
            fail(f"entity_name: not unique ({len(set(entity_names))} unique for {n} rows)")

    empty_data = sum(1 for x in data_values if not x)
    if empty_data == 0:
        ok(f"data: all non-empty ({n} rows)")
    else:
        fail(f"data: {empty_data} empty, {n - empty_data} non-empty")

    n_with_image = len(image_ids)
    n_empty_image = n - n_with_image
    unique_image = len(set(image_ids)) == n_with_image
    if unique_image:
        ok(f"image_id: unique among non-empty ({n_with_image} non-empty)")
    else:
        fail(f"image_id: not unique ({len(set(image_ids))} unique for {n_with_image} non-empty)")
    print(f"  image_id stats: {n_with_image} non-empty, {n_empty_image} empty (total {n})")


def main():
    try:
        csv.field_size_limit(2**31 - 1)
    except OverflowError:
        csv.field_size_limit(2**20)

    if not DATA_CSV.exists():
        print(f"Not found: {DATA_CSV}")
        return

    fieldnames = ["entity_id", "entity_name", "data", "image_id"]

    # Read all rows
    print("Reading CSV ...")
    rows = []
    with open(DATA_CSV, "r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["entity_id"] = (row.get("entity_id") or "").strip()
            row["entity_name"] = (row.get("entity_name") or "").strip()
            row["data"] = row.get("data") or ""
            row["image_id"] = (row.get("image_id") or "").strip()
            rows.append(row)

    n_before = len(rows)
    print(f"Loaded {n_before} rows. Deduplicating by entity_name (keep one with non-empty image_id if any) ...")

    # Keep one row per entity_name: prefer non-empty image_id, else first
    by_name = {}
    for row in rows:
        name = row["entity_name"]
        if name not in by_name:
            by_name[name] = row
        else:
            existing = by_name[name]
            # Prefer row that has image_id
            if (row["image_id"] and not existing["image_id"]):
                by_name[name] = row

    kept = list(by_name.values())
    n_after = len(kept)
    # Re-assign entity_id to e1, e2, ...
    for i, row in enumerate(kept):
        row["entity_id"] = f"e{i + 1}"

    print(f"Kept {n_after} rows (removed {n_before - n_after} duplicates). Writing back to CSV ...")

    with open(DATA_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print()
    print("Health check: data/yelp_dataset.csv (after dedup)")
    print("=" * 50)
    run_health_check(kept)
    print("=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
