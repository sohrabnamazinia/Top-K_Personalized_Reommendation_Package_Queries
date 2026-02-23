from dataclasses import dataclass
from typing import List, Set, Optional


@dataclass
class Entity:
    """Represents an entity with id, name, data, and optional image_id for vision."""
    id: str
    name: str
    data: str
    image_id: Optional[str] = None


@dataclass
class Component:
    """Represents a component with name, description, and dimension."""
    name: str
    description: str
    dimension: int  # 1 for unary, 2 for binary


@dataclass
class Package:
    """Represents a package (set) of entities."""
    entities: Set[str]  # Set of entity IDs

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self.entities

    def size(self) -> int:
        return len(self.entities)

    def add(self, entity_id: str):
        self.entities.add(entity_id)

    def remove(self, entity_id: str):
        self.entities.discard(entity_id)

    def copy(self) -> 'Package':
        return Package(entities=self.entities.copy())
