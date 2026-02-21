"""
Build hotels_dataset.csv from hotels.csv.

Output: data/hotels_dataset.csv with columns entity_id, entity_name, data.
- entity_id: e1, e2, e3, ...
- entity_name: from HotelName column
- data: bundle of HotelName, cityName, HotelRating, Address, Description (one string per row).

Set N_ENTITIES below (e.g. 5 for testing). None = no limit (process entire file).
"""
import csv
from pathlib import Path

# ---------- Parameters ----------
ROOT = Path(__file__).parent.parent
HOTELS_CSV = ROOT / "data" / "hotels.csv"
OUTPUT_CSV = ROOT / "data" / "hotels_dataset.csv"
N_ENTITIES = 20000   # None = all rows; set to int (e.g. 5 or 20000) to limit

# Columns to bundle into data (must exist in hotels.csv)
DATA_COLUMNS = ["HotelName", "cityName", "HotelRating", "Address", "Description"]
# ---------------------------------

PROGRESS_EVERY = 2000


def main():
    if not HOTELS_CSV.exists():
        print(f"File not found: {HOTELS_CSV}")
        return

    n = N_ENTITIES
    print(f"Processing hotels from {HOTELS_CSV} -> {OUTPUT_CSV}")
    if n is not None:
        print(f"Limit: first {n} rows")
    else:
        print("Limit: none (all rows)")
    print("...")

    count = 0
    with open(HOTELS_CSV, "r", encoding="utf-8", errors="replace") as fin, open(
        OUTPUT_CSV, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["entity_id", "entity_name", "data"])
        writer.writeheader()
        for i, row in enumerate(reader):
            if n is not None and i >= n:
                break
            # Normalize keys (CSV has spaces after commas, e.g. " HotelName" not "HotelName")
            row = {k.strip(): v for k, v in row.items()}
            entity_id = f"e{i + 1}"
            entity_name = (row.get("HotelName") or "").strip() or f"Hotel {i + 1}"
            data_parts = [f"{col}: {row.get(col, '')}" for col in DATA_COLUMNS]
            data = "\n".join(data_parts)
            writer.writerow({"entity_id": entity_id, "entity_name": entity_name, "data": data})
            count += 1
            if count % PROGRESS_EVERY == 0:
                print(f"  {count} rows written ...")

    print(f"Done. Wrote {count} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
