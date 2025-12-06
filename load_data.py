"""Utility to load entities from CSV file."""
import csv
from typing import Dict
from models import Entity


def load_entities_from_csv(csv_path: str) -> Dict[str, Entity]:
    """
    Load entities from CSV file.
    
    Expected CSV format:
    entity_id,entity_name,data
    
    Returns:
        Dictionary mapping entity_id to Entity
    """
    entities = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entity_id = row['entity_id'].strip()
            entity_name = row['entity_name'].strip()
            data = row['data'].strip()
            
            entities[entity_id] = Entity(
                id=entity_id,
                name=entity_name,
                data=data
            )
    
    return entities

