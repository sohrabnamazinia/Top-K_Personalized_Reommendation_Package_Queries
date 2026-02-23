"""Utility to load entities from CSV file."""
import csv
from typing import Dict
from utils.models import Entity


def load_entities_from_csv(csv_path: str) -> Dict[str, Entity]:
    """
    Load entities from CSV file.

    Expected CSV format: entity_id, entity_name, data [, image_id]
    If image_id column is present it is loaded; otherwise Entity.image_id is None.

    Returns:
        Dictionary mapping entity_id to Entity
    """
    try:
        csv.field_size_limit(2**31 - 1)
    except OverflowError:
        csv.field_size_limit(2**20)

    entities = {}
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_image_id = 'image_id' in fieldnames
        for row in reader:
            entity_id = (row.get('entity_id') or '').strip()
            entity_name = (row.get('entity_name') or '').strip()
            data = (row.get('data') or '').strip()
            image_id = (row.get('image_id') or '').strip() or None if has_image_id else None
            entities[entity_id] = Entity(
                id=entity_id,
                name=entity_name,
                data=data,
                image_id=image_id,
            )
    return entities
