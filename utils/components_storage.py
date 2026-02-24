"""
Central storage for scoring functions F1–F6 (query + components) and their dataset paths.
Used by run_MGT scripts and scalability experiments.
"""
from pathlib import Path
from typing import Dict, List, Optional

from .models import Component

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# F1, F2: Hotels
# ---------------------------------------------------------------------------
F1 = {
    "query": "Find hotels where each hotel balances comfort, safety, and accessibility, and that collectively span different price tiers.",
    "components": [
        Component(name="c1", description="Hotels that each balances comfort, safety, and accessibility", dimension=1),
        Component(name="c2", description="Hotels spanning different price tiers", dimension=2),
    ],
}

F2 = {
    "query": "Find hotels near cultural and academic hubs that offer a quiet, work-friendly environment and are close to public transportation, while ensuring a diverse mix of large hotel brands, boutique hotels, and local properties.",
    "components": [
        Component(name="c1", description="Hotels that are near cultural and academic hubs, and offer a quiet work-friendly environment", dimension=1),
        Component(name="c2", description="Hotels that are close to public transportation", dimension=1),
        Component(name="c3", description="Diverse hotels, ensuring a mix of large hotel brands, boutique hotels, and local properties", dimension=2),
    ],
}

# ---------------------------------------------------------------------------
# F3, F4: Movies
# ---------------------------------------------------------------------------
F3 = {
    "query": "package of high-quality movies about space exploration and astronauts that feel realistic and scientifically grounded. The package should be diverse in tone and perspective (e.g., survival-focused, philosophical, political, emotional), so the movies don't all tell the same kind of story.",
    "components": [
        Component(name="c1", description="High quality movies about space exploration and astronauts with realistic or science-driven themes.", dimension=1),
        Component(name="c2", description="Diverse in tone and perspective (e.g., survival-focused, philosophical, political, emotional), so the movies don't all tell the same kind of story.", dimension=2),
    ],
}

F4 = {
    "query": "Intellectually engaging and character-driven movies that could be watched with family during holidays. The movies should have diverse cultural perspectives and length",
    "components": [
        Component(name="c1", description="Intellectually engaging movies", dimension=1),
        Component(name="c2", description="Character-driven movies", dimension=1),
        Component(name="c3", description="Movies that could be watched with family during holidays", dimension=1),
        Component(name="c4", description="Movies with diverse cultural perspectives and length", dimension=2),
    ],
}

# ---------------------------------------------------------------------------
# F5, F6: Yelp
# ---------------------------------------------------------------------------
F5 = {
    "query": "Minority-owned restaurants, with highly rated reviews from food experts",
    "components": [
        Component(name="c1", description="Minority-owned restaurants", dimension=1),
        Component(name="c2", description="Restaurants with highly rated reviews from food experts", dimension=1),
    ],
}

F6 = {
    "query": "Find restaurants in a nice neighborhood suitable for family dinner that provides different cuisines and price range",
    "components": [
        Component(name="c1", description="Restaurants in a nice neighborhood", dimension=1),
        Component(name="c2", description="Restaurants suitable for family dinner", dimension=1),
        Component(name="c3", description="Restaurants that provide different cuisines and price range", dimension=2),
    ],
}

# ---------------------------------------------------------------------------
# Config lookup and dataset paths
# ---------------------------------------------------------------------------
CONFIGS: Dict[str, dict] = {
    "f1": F1,
    "f2": F2,
    "f3": F3,
    "f4": F4,
    "f5": F5,
    "f6": F6,
}

# Dataset path for each scoring function (relative to project root)
DATASET_PATH: Dict[str, Path] = {
    "f1": ROOT / "data" / "hotels_dataset.csv",
    "f2": ROOT / "data" / "hotels_dataset.csv",
    "f3": ROOT / "data" / "movies_dataset.csv",
    "f4": ROOT / "data" / "movies_dataset.csv",
    "f5": ROOT / "data" / "yelp_dataset.csv",
    "f6": ROOT / "data" / "yelp_dataset.csv",
}


def get_config(scoring: str) -> dict:
    """Return {query, components} for scoring function f1–f6."""
    if scoring.lower() not in CONFIGS:
        raise ValueError(f"Unknown scoring: {scoring}. Use one of {list(CONFIGS.keys())}")
    return CONFIGS[scoring.lower()]


def get_dataset_path(scoring: str) -> Path:
    """Return path to CSV dataset for the given scoring function."""
    if scoring.lower() not in DATASET_PATH:
        raise ValueError(f"Unknown scoring: {scoring}")
    return DATASET_PATH[scoring.lower()]
