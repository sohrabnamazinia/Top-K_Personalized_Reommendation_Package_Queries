"""
Measure token usage distributions for two "probe" types:
  1) text-only probe (e.g., f1)
  2) text+image probe (e.g., f5/f6 with Yelp photos)

Runs each probe type N_TRIALS times using REAL LLM calls (unless you change MOCK_API=True).
Prints mean/std for prompt/completion/total tokens and estimated $ cost per call under:
  - gpt-4o-mini input rate ($0.15 / 1M input tokens)
  - gpt-4o      input rate ($2.50 / 1M input tokens)

Notes:
- OpenAI pricing provided by you is INPUT tokens only, so cost uses prompt_tokens.
- Token usage is taken from the LLM response usage metadata (no tiktoken needed).

Run from project root:
  python measure_llm_token_costs.py
"""

from __future__ import annotations

import base64
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path
from preprocessing.load_data import load_entities_from_csv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# -----------------------------
# CONFIG (edit here)
# -----------------------------
MODEL_FOR_MEASUREMENT = "gpt-4o-mini"
N_TRIALS = 10

# If True, uses random outputs (no real LLM) => token usage will be unavailable.
MOCK_API = False

# Pricing (INPUT tokens only), $ per 1M tokens
PRICE_PER_1M_INPUT_TOKENS = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
}

# Text-only probe source
TEXT_SCORING = "f1"
TEXT_COMPONENT_NAME = "c1"  # unary

# Text+image probe source (Yelp)
IMAGE_SCORING = "f5"
IMAGE_COMPONENT_NAME = "c1"  # unary
YELP_PHOTOS_DIR = ROOT / "data" / "yelp_dataset" / "photos"

# If you don't have Yelp photos locally, we simulate "image payload" by embedding a base64 blob in text.
# This increases prompt tokens and provides a controllable proxy cost distribution.
SIMULATE_IMAGE_BASE64_IN_TEXT_IF_NO_FILES = True
SIMULATED_IMAGE_BYTES = 20_000  # 20KB raw bytes -> ~27KB base64 chars (increase/decrease as needed)


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _extract_usage(resp: Any) -> Optional[Usage]:
    """
    Best-effort extraction of token usage from a LangChain AIMessage.
    """
    # Newer LangChain often provides `usage_metadata`
    usage = getattr(resp, "usage_metadata", None)
    if isinstance(usage, dict):
        pt = usage.get("input_tokens") or usage.get("prompt_tokens")
        ct = usage.get("output_tokens") or usage.get("completion_tokens")
        tt = usage.get("total_tokens")
        if pt is not None and ct is not None:
            pt_i, ct_i = int(pt), int(ct)
            tt_i = int(tt) if tt is not None else (pt_i + ct_i)
            return Usage(prompt_tokens=pt_i, completion_tokens=ct_i, total_tokens=tt_i)

    # Older path: response_metadata.token_usage
    rm = getattr(resp, "response_metadata", None)
    if isinstance(rm, dict):
        tu = rm.get("token_usage") or rm.get("usage") or rm.get("tokenUsage")
        if isinstance(tu, dict):
            pt = tu.get("prompt_tokens") or tu.get("input_tokens")
            ct = tu.get("completion_tokens") or tu.get("output_tokens")
            tt = tu.get("total_tokens")
            if pt is not None and ct is not None:
                pt_i, ct_i = int(pt), int(ct)
                tt_i = int(tt) if tt is not None else (pt_i + ct_i)
                return Usage(prompt_tokens=pt_i, completion_tokens=ct_i, total_tokens=tt_i)

    return None


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (float(statistics.mean(xs)), float(statistics.pstdev(xs)))


def _escape_braces(s: str) -> str:
    return (s or "").replace("{", "{{").replace("}", "}}")


MAX_ENTITY_DATA_CHARS = 10_000


def _truncate_data(s: str) -> str:
    if not s or len(s) <= MAX_ENTITY_DATA_CHARS:
        return s or ""
    return s[:MAX_ENTITY_DATA_CHARS] + "\n\n[... truncated for context length ...]"


def _image_data_url(image_id: str) -> Optional[str]:
    if not image_id:
        return None
    base_path = Path(YELP_PHOTOS_DIR)
    for ext in (".jpg", ".jpeg", ".png", ""):
        img_path = (base_path / f"{image_id}{ext}") if ext else (base_path / image_id)
        if img_path.is_file():
            b64 = base64.standard_b64encode(img_path.read_bytes()).decode("ascii")
            mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            return f"data:{mime};base64,{b64}"
    return None


def _find_yelp_entities_with_images(entities: Dict[str, Any], n: int) -> List[str]:
    out: List[str] = []
    for eid, ent in entities.items():
        image_id = getattr(ent, "image_id", None)
        if not image_id:
            continue
        if _image_data_url(str(image_id)) is None:
            continue
        out.append(eid)
        if len(out) >= n:
            break
    if not out:
        raise FileNotFoundError(f"No Yelp entities with images found under {YELP_PHOTOS_DIR}")
    while len(out) < n:
        out.append(out[len(out) % len(out)])
    return out


def _build_messages_text_only(*, component, entities: Dict[str, Any], entity_id: str, query: str) -> List[Any]:
    ent = entities[entity_id]
    data_trunc = _truncate_data(getattr(ent, "data", ""))

    system_prompt = f"""You are part of a top-k retrieval system for multimodal data, formulated as a package query problem.

Your specific role is to evaluate ONE component value, which we call "{component.name}".

Component Details:
- Component Name: {component.name}
- Component Description: {_escape_braces(component.description)}
- Component Dimension: {component.dimension} ({'unary' if component.dimension == 1 else 'binary'})

CRITICAL REQUIREMENTS:
1. Return exactly: lower_bound, upper_bound
2. Both in [0, 1] and MUST be exactly the same (e.g., "0.5, 0.5")
3. Return ONLY the two numbers separated by a comma (no other text)
"""

    human_prompt = f"""User Query: {_escape_braces(query)}

Entity Information:
Entity 1 (ID: {_escape_braces(getattr(ent, 'id', entity_id))}, Name: {_escape_braces(getattr(ent, 'name', ''))}, Data: {_escape_braces(data_trunc)})

Evaluate the {component.name} component value. Return only two float numbers (lower_bound, upper_bound) separated by a comma, both in range [0, 1]. IMPORTANT: Both values must be EXACTLY THE SAME (e.g., "0.5, 0.5"):"""

    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def _build_messages_text_plus_image(*, component, entities: Dict[str, Any], entity_id: str, query: str) -> List[Any]:
    ent = entities[entity_id]
    data_trunc = _truncate_data(getattr(ent, "data", ""))
    image_id = getattr(ent, "image_id", None)
    url = _image_data_url(str(image_id)) if image_id else None

    system_prompt = f"""You are part of a top-k retrieval system for multimodal data, formulated as a package query problem.

Your specific role is to evaluate ONE component value, which we call "{component.name}".

Component Details:
- Component Name: {component.name}
- Component Description: {_escape_braces(component.description)}
- Component Dimension: {component.dimension} (unary)

CRITICAL REQUIREMENTS:
1. Return exactly: lower_bound, upper_bound
2. Both in [0, 1] and MUST be exactly the same (e.g., "0.5, 0.5")
3. Return ONLY the two numbers separated by a comma (no other text)
"""

    simulated_image_note = ""
    if url is None and SIMULATE_IMAGE_BASE64_IN_TEXT_IF_NO_FILES:
        # Deterministic base64 blob to simulate an "image payload" in the prompt.
        # (This is not sent as an actual image; it's plain text to affect token usage.)
        b64 = base64.standard_b64encode(b"\x00" * int(SIMULATED_IMAGE_BYTES)).decode("ascii")
        simulated_image_note = (
            "\n\n[Simulated image payload (base64) begins]\n"
            f"{b64}\n"
            "[Simulated image payload (base64) ends]\n"
        )

    human_text = f"""User Query: {_escape_braces(query)}

Entity Information:
Entity 1 (ID: {_escape_braces(getattr(ent, 'id', entity_id))}, Name: {_escape_braces(getattr(ent, 'name', ''))}, Data: {_escape_braces(data_trunc)})
{simulated_image_note}

Evaluate the {component.name} component value. Return only two float numbers (lower_bound, upper_bound) separated by a comma, both in range [0, 1]. IMPORTANT: Both values must be EXACTLY THE SAME (e.g., "0.5, 0.5"):"""

    # If we have a real image file, send as multimodal; else use text-only with simulated base64.
    if url is not None:
        content = [
            {"type": "text", "text": human_text},
            {"type": "image_url", "image_url": {"url": url}},
        ]
        return [SystemMessage(content=system_prompt), HumanMessage(content=content)]
    return [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]


def _find_yelp_entity_with_image(entities: Dict[str, Any]) -> str:
    # Backward-compatible helper (returns the first)
    return _find_yelp_entities_with_images(entities, 1)[0]


def _get_component(components: list, name: str):
    for c in components:
        if c.name == name:
            return c
    raise ValueError(f"Component not found: {name}. Available: {[c.name for c in components]}")


def _run_probe_trials(
    *,
    title: str,
    llm: ChatOpenAI,
    entities: Dict[str, Any],
    query: str,
    component,
    entity_ids: List[str],
    n_trials: int,
    with_image: bool,
) -> Dict[str, Any]:
    prompt_tokens: List[int] = []
    completion_tokens: List[int] = []
    total_tokens: List[int] = []

    for i in range(n_trials):
        eid = entity_ids[i % len(entity_ids)]
        messages = (
            _build_messages_text_plus_image(component=component, entities=entities, entity_id=eid, query=query)
            if with_image
            else _build_messages_text_only(component=component, entities=entities, entity_id=eid, query=query)
        )
        resp = llm.invoke(messages)
        usage = _extract_usage(resp)
        if usage is None:
            raise RuntimeError(
                "Could not extract token usage from LLM response.\n"
                f"response_metadata keys: {list((getattr(resp, 'response_metadata', None) or {}).keys())}\n"
                f"usage_metadata: {getattr(resp, 'usage_metadata', None)}"
            )

        prompt_tokens.append(usage.prompt_tokens)
        completion_tokens.append(usage.completion_tokens)
        total_tokens.append(usage.total_tokens)

    pt_mean, pt_std = _mean_std([float(x) for x in prompt_tokens])
    ct_mean, ct_std = _mean_std([float(x) for x in completion_tokens])
    tt_mean, tt_std = _mean_std([float(x) for x in total_tokens])

    cost_estimates = {}
    for model, price_per_1m in PRICE_PER_1M_INPUT_TOKENS.items():
        # input tokens only
        mean_cost = (pt_mean / 1_000_000.0) * price_per_1m
        std_cost = (pt_std / 1_000_000.0) * price_per_1m
        cost_estimates[model] = {"mean_cost_usd": mean_cost, "std_cost_usd": std_cost}

    return {
        "title": title,
        "n_trials": n_trials,
        "prompt_tokens": {"mean": pt_mean, "std": pt_std, "min": min(prompt_tokens), "max": max(prompt_tokens)},
        "completion_tokens": {"mean": ct_mean, "std": ct_std, "min": min(completion_tokens), "max": max(completion_tokens)},
        "total_tokens": {"mean": tt_mean, "std": tt_std, "min": min(total_tokens), "max": max(total_tokens)},
        "cost_input_only_usd": cost_estimates,
    }


def main() -> None:
    if MOCK_API:
        raise SystemExit("Set MOCK_API=False to do real calls and get usage from the provider.")

    # Text-only setup (hotels)
    text_cfg = get_config(TEXT_SCORING)
    text_query = text_cfg["query"]
    text_components = text_cfg["components"]
    text_dataset = get_dataset_path(TEXT_SCORING)
    text_entities = load_entities_from_csv(str(text_dataset))
    text_entity_ids = list(text_entities.keys())[: max(1, N_TRIALS)]
    text_comp = _get_component(text_components, TEXT_COMPONENT_NAME)

    # Image setup (yelp)
    img_cfg = get_config(IMAGE_SCORING)
    img_query = img_cfg["query"]
    img_components = img_cfg["components"]
    img_dataset = get_dataset_path(IMAGE_SCORING)
    img_entities = load_entities_from_csv(str(img_dataset))
    try:
        img_entity_ids = _find_yelp_entities_with_images(img_entities, N_TRIALS)
    except FileNotFoundError:
        if not SIMULATE_IMAGE_BASE64_IN_TEXT_IF_NO_FILES:
            raise
        # Fall back to any entities; we'll simulate the image payload in text.
        img_entity_ids = list(img_entities.keys())[: max(1, N_TRIALS)]
    img_comp = _get_component(img_components, IMAGE_COMPONENT_NAME)

    llm_text = ChatOpenAI(model=MODEL_FOR_MEASUREMENT, temperature=0)
    llm_img = ChatOpenAI(model=MODEL_FOR_MEASUREMENT, temperature=0)

    print("=" * 80)
    print("Token usage measurement")
    print("=" * 80)
    print(f"model_for_measurement={MODEL_FOR_MEASUREMENT}  n_trials={N_TRIALS}")
    print("pricing_input_only_per_1M_tokens:", PRICE_PER_1M_INPUT_TOKENS)
    if SIMULATE_IMAGE_BASE64_IN_TEXT_IF_NO_FILES:
        print(f"simulate_image_base64_if_missing=True  simulated_image_bytes={SIMULATED_IMAGE_BYTES}")
    print()

    r1 = _run_probe_trials(
        title="text_only",
        llm=llm_text,
        entities=text_entities,
        query=text_query,
        component=text_comp,
        entity_ids=text_entity_ids,
        n_trials=N_TRIALS,
        with_image=False,
    )
    r2 = _run_probe_trials(
        title="text_plus_image",
        llm=llm_img,
        entities=img_entities,
        query=img_query,
        component=img_comp,
        entity_ids=img_entity_ids,
        n_trials=N_TRIALS,
        with_image=True,
    )

    for r in (r1, r2):
        print("-" * 80)
        print(r["title"])
        print("prompt_tokens:", r["prompt_tokens"])
        print("completion_tokens:", r["completion_tokens"])
        print("total_tokens:", r["total_tokens"])
        print("cost_input_only_usd:", r["cost_input_only_usd"])


if __name__ == "__main__":
    main()

