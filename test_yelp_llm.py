"""
Simple test: load two entities from yelp_dataset.csv and evaluate a degree-1 component
with the LLM (text + image when image_id is present).
Run from project root: python test_yelp_llm.py
Requires OPENAI_API_KEY and a vision-capable model (e.g. gpt-4o-mini).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.models import Component, Entity
from utils.llm_interface import LLMEvaluator
from preprocessing.load_data import load_entities_from_csv


def main():
    csv_path = ROOT / "data" / "yelp_dataset.csv"
    photos_dir = str(ROOT / "data" / "yelp_dataset" / "photos")

    if not csv_path.exists():
        print(f"Not found: {csv_path}")
        return

    entities_all = load_entities_from_csv(str(csv_path))
    # Take two entities that have non-empty image_id
    with_image = [(eid, e) for eid, e in entities_all.items() if getattr(e, "image_id", None)]
    if len(with_image) < 2:
        print("Need at least 2 entities with non-empty image_id in the CSV.")
        return
    entity_ids = [with_image[0][0], with_image[1][0]]
    entities = {eid: entities_all[eid] for eid in entity_ids}

    print("Entities:")
    for eid in entity_ids:
        e = entities[eid]
        print(f"  {eid}: {e.name!r}  image_id={getattr(e, 'image_id', None)!r}")

    component = Component(
        name="quality",
        description="Overall quality and appeal of the business based on reviews and appearance.",
        dimension=1,
    )
    query = "Find restaurants or cafes that are high quality and appealing."

    evaluator = LLMEvaluator(
        mock_api=False,
        use_MGT=False,
        images_base_path=photos_dir,
    )

    print(f"\nEvaluating component {component.name!r} (dim 1) with query: {query!r}")
    print("(Entity with image_id will get image sent to LLM.)\n")

    for eid in entity_ids:
        lb, ub, t = evaluator.evaluate_component(
            component, entities, [eid], query, use_cache=False
        )
        print(f"  {eid} ({entities[eid].name}): ({lb}, {ub})  time={t}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
