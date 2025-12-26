"""
NER quality scoring utilities.

Purpose:
- Convert structured NER output into a confidence-like score (0.0 → 1.0)
- Decide whether NER output is reliable or needs LLM repair
"""

from typing import Dict, Any, List


def score_ner(entities: Dict[str, Any], intent: str | None = None) -> float:
    """
    Compute a heuristic quality score for NER output.

    Parameters
    ----------
    entities : dict
        Output of postprocess_ner()
    intent : str | None
        Final intent (ADD_ITEM, REMOVE_ITEM, etc.)

    Returns
    -------
    float
        Score between 0.0 (bad) and 1.0 (excellent)
    """

    # -----------------------------
    # Basic validation
    # -----------------------------
    if not entities or not isinstance(entities, dict):
        return 0.0

    score = 1.0

    # -----------------------------
    # Food items
    # -----------------------------
    food_items: List[str] = entities.get("food_items", [])
    if not food_items:
        score -= 0.4

    # -----------------------------
    # Quantity
    # -----------------------------
    quantity = entities.get("quantity")
    if intent in {"ADD_ITEM", "REMOVE_ITEM"}:
        if quantity is None:
            score -= 0.2
        elif isinstance(quantity, int) and quantity <= 0:
            score -= 0.2

    # -----------------------------
    # Ambiguity
    # -----------------------------
    clarifications = entities.get("clarification", [])
    if clarifications:
        # penalize ambiguity but do not kill score
        score -= min(0.3, 0.1 * len(clarifications))

    # -----------------------------
    # Unknown items (optional key)
    # -----------------------------
    unknown_items = entities.get("unknown_items", [])
    if unknown_items:
        score -= min(0.3, 0.1 * len(unknown_items))

    # -----------------------------
    # Sanity clamp
    # -----------------------------
    return round(max(0.0, min(score, 1.0)), 2)
