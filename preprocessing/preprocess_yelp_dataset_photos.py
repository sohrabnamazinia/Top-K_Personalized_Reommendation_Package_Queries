"""
Fill image_id column in data/yelp_dataset.csv using photos.json and business JSON.

For each row in yelp_dataset.csv: look up business_id by entity_name (business name) from
yelp_academic_dataset_business.json; then in photos.json pick one photo whose business_id
matches; set image_id to that photo_id.
"""
import json
import csv
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).parent.parent
YELP_DIR = ROOT / "data" / "yelp_dataset"
DATA_CSV = ROOT / "data" / "yelp_dataset.csv"
BUSINESS_JSON = YELP_DIR / "yelp_academic_dataset_business.json"
PHOTOS_JSON = YELP_DIR / "photos.json"
# ---------------------------------
PROGRESS_BUSINESS = 30_000   # print every N lines when loading business JSON
PROGRESS_PHOTOS = 100_000   # print every N lines when loading photos JSON
PROGRESS_CSV = 1_000        # print every N rows when reading CSV


def main():
    # Allow very large fields (e.g. data column with bundled reviews)
    try:
        csv.field_size_limit(2**31 - 1)
    except OverflowError:
        csv.field_size_limit(2**20)  # fallback ~1MB

    if not DATA_CSV.exists():
        print(f"Not found: {DATA_CSV}. Run preprocess_yelp_dataset.py first.")
        return
    if not BUSINESS_JSON.exists() or not PHOTOS_JSON.exists():
        print(f"Not found: {BUSINESS_JSON} or {PHOTOS_JSON}")
        return

    print("Loading business name -> business_id from business JSON ...")
    name_to_bid = {}
    line_count = 0
    with open(BUSINESS_JSON, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            if line_count % PROGRESS_BUSINESS == 0:
                print(f"  Business JSON: {line_count} lines read, {len(name_to_bid)} names so far ...")
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name = (obj.get("name") or "").strip()
            if not name:
                continue
            bid = obj.get("business_id")
            if bid and name not in name_to_bid:
                name_to_bid[name] = bid
    print(f"  Loaded {len(name_to_bid)} business names (from {line_count} lines).")

    print("Loading business_id -> one photo_id from photos.json ...")
    bid_to_photo = {}
    line_count = 0
    with open(PHOTOS_JSON, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            if line_count % PROGRESS_PHOTOS == 0:
                print(f"  Photos JSON: {line_count} lines read, {len(bid_to_photo)} businesses with photo so far ...")
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            bid = obj.get("business_id")
            pid = obj.get("photo_id")
            if bid and pid and bid not in bid_to_photo:
                bid_to_photo[bid] = pid
    print(f"  Loaded {len(bid_to_photo)} businesses with at least one photo (from {line_count} lines).")

    print(f"Reading {DATA_CSV} and filling image_id ...")
    rows = []
    with open(DATA_CSV, "r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader):
            if i > 0 and i % PROGRESS_CSV == 0:
                print(f"  CSV rows processed: {i} ...")
            entity_name = (row.get("entity_name") or "").strip()
            bid = name_to_bid.get(entity_name)
            photo_id = bid_to_photo.get(bid, "") if bid else ""
            row["image_id"] = photo_id
            rows.append(row)
    print(f"  Read {len(rows)} rows.")

    print(f"Writing {DATA_CSV} ...")
    with open(DATA_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    filled = sum(1 for r in rows if r.get("image_id"))
    print(f"Done. Filled image_id for {filled} / {len(rows)} rows.")


if __name__ == "__main__":
    main()
