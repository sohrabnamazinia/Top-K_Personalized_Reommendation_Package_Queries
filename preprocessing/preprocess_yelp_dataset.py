"""
Build yelp_dataset.csv from Yelp academic JSON (businesses + reviews).

Output: data/yelp_dataset.csv with columns entity_id, entity_name, data, image_id.
- entity_id: e1, e2, e3, ...
- entity_name: from "name" in business JSON
- data: bundle of "stars" and "text" from all reviews for that business (reviews linked by business_id)
- image_id: empty for now

Set N_ENTITIES below (e.g. 5 for testing). None = process all businesses in business file.
"""
import json
import csv
from pathlib import Path

# ---------- Parameters ----------
ROOT = Path(__file__).parent.parent
YELP_DIR = ROOT / "data" / "yelp_dataset"
BUSINESS_JSON = YELP_DIR / "yelp_academic_dataset_business.json"
REVIEW_JSON = YELP_DIR / "yelp_academic_dataset_review.json"
OUTPUT_CSV = ROOT / "data" / "yelp_dataset.csv"
N_ENTITIES = 40000   # None = all businesses; set to int to limit

PROGRESS_EVERY = 5000
# ---------------------------------


def main():
    if not BUSINESS_JSON.exists() or not REVIEW_JSON.exists():
        print(f"Required files not found in {YELP_DIR}")
        return

    n = N_ENTITIES
    print(f"Loading first {n or 'all'} businesses from {BUSINESS_JSON} ...")
    # First pass: load business_id -> name for first n businesses (order preserved)
    business_order = []
    business_names = {}
    with open(BUSINESS_JSON, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            bid = obj.get("business_id")
            name = (obj.get("name") or "").strip() or f"Business {len(business_order) + 1}"
            business_order.append(bid)
            business_names[bid] = name
            if n is not None and len(business_order) >= n:
                break

    business_ids_set = set(business_order)
    print(f"Loaded {len(business_order)} businesses. Scanning reviews for those businesses ...")

    # Second pass: collect reviews (stars, text) per business_id only for our n businesses
    reviews_by_business = {bid: [] for bid in business_order}
    with open(REVIEW_JSON, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i > 0 and i % PROGRESS_EVERY == 0:
                print(f"  Reviews scanned: {i} ...")
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            bid = obj.get("business_id")
            if bid not in business_ids_set:
                continue
            stars = obj.get("stars", "")
            text = (obj.get("text") or "").strip()
            reviews_by_business[bid].append((stars, text))

    print(f"Writing {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_id", "entity_name", "data", "image_id"])
        for i, bid in enumerate(business_order):
            entity_id = f"e{i + 1}"
            entity_name = business_names[bid]
            parts = []
            for stars, text in reviews_by_business[bid]:
                parts.append(f"stars: {stars}\ntext: {text}")
            data = "\n\n".join(parts) if parts else ""
            writer.writerow([entity_id, entity_name, data, ""])

    print(f"Done. Wrote {len(business_order)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
