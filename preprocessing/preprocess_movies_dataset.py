"""
Temporary script to build movies_dataset.csv from movie_plots.csv.

Output: data/movies_dataset.csv with columns entity_id, entity_name, data.
- entity_id: e1, e2, e3, ...
- entity_name: Movie 1, Movie 2, ...
- data: one string per row with "attr: value\\n" for each source column.

Set N_ENTITIES below (e.g. 5 for testing, 20000 for full run). Cap is 20,000.
"""
import csv
from pathlib import Path

# ---------- Parameters ----------
ROOT = Path(__file__).parent.parent
MOVIE_PLOTS_CSV = ROOT / "data" / "movie_plots.csv"
OUTPUT_CSV = ROOT / "data" / "movies_dataset.csv"
N_ENTITIES = 20000   # Test with 5; set to 20000 for full dataset (max 20,000)
MAX_ENTITIES = 20_000
# ---------------------------------


PROGRESS_EVERY = 2000  # print progress every N rows


def main():
    n = min(N_ENTITIES, MAX_ENTITIES)
    if N_ENTITIES > MAX_ENTITIES:
        print(f"N_ENTITIES capped from {N_ENTITIES} to {MAX_ENTITIES}")
    print(f"Processing up to {n} rows, writing to {OUTPUT_CSV} ...")

    count = 0
    with open(MOVIE_PLOTS_CSV, "r", encoding="utf-8") as fin, open(
        OUTPUT_CSV, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        writer = csv.DictWriter(fout, fieldnames=["entity_id", "entity_name", "data"])
        writer.writeheader()
        for i, row in enumerate(reader):
            if i >= n:
                break
            entity_id = f"e{i + 1}"
            entity_name = f"Movie {i + 1}"
            data_parts = [f"{k}: {row.get(k, '')}" for k in fieldnames]
            data = "\n".join(data_parts)
            writer.writerow({"entity_id": entity_id, "entity_name": entity_name, "data": data})
            count += 1
            if count % PROGRESS_EVERY == 0:
                print(f"  {count} rows written ...")

    print(f"Done. Wrote {count} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
