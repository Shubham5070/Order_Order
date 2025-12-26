import json
import re
from rapidfuzz import process, fuzz
from typing import Dict, Any, List

# -------------------------------
# Constants
# -------------------------------
WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

GENERIC_HEADS = {"naan", "dosa", "paneer", "paratha", "sandwich", "roti"}


# -------------------------------
# Load menu items
# -------------------------------
def load_menu_items(menu_path: str) -> List[str]:
    with open(menu_path, "r", encoding="utf-8") as f:
        menu = json.load(f)

    items = []
    menus = menu if isinstance(menu, list) else [menu]

    for m in menus:
        for it in m.get("items", []):
            if it.get("name"):
                items.append(it["name"].strip().lower())
            for a in it.get("aliases", []):
                items.append(a.strip().lower())

    return list(set(items))


# -------------------------------
# Quantity normalization
# -------------------------------
def normalize_quantity(text: str):
    text = text.lower().strip()
    if text.isdigit():
        return int(text)
    return WORD_TO_NUM.get(text)


# -------------------------------
# Food item normalization
# -------------------------------
def normalize_food_item(raw: str, menu_items: list, threshold: int = 80):
    raw = raw.lower().strip()
    raw = re.sub(r"[^\w\s]", "", raw)

    # Exact
    if raw in menu_items:
        return raw

    # Singular
    if raw.endswith("s") and raw[:-1] in menu_items:
        return raw[:-1]

    # Generic head → ALWAYS ambiguous
    if raw in GENERIC_HEADS:
        return {
            "ambiguous": raw,
            "options": [m for m in menu_items if m.endswith(raw)]
        }

    # Partial match
    partials = [m for m in menu_items if m.startswith(raw + " ")]
    if len(partials) == 1:
        return partials[0]
    if len(partials) > 1:
        return {"ambiguous": raw, "options": partials}

    # Fuzzy → ambiguity only
    result = process.extractOne(raw, GENERIC_HEADS, scorer=fuzz.ratio)
    if result and result[1] >= threshold:
        head = result[0]
        return {
            "ambiguous": raw,
            "options": [m for m in menu_items if m.endswith(head)]
        }

    return None


# -------------------------------
# MAIN NER POSTPROCESSOR
# -------------------------------
def postprocess_ner(doc, menu_items: list) -> Dict[str, Any]:
    items = []
    clarifications = []
    last_qty = None

    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            last_qty = normalize_quantity(ent.text)

        elif ent.label_ == "FOOD_ITEM":
            normalized = normalize_food_item(ent.text, menu_items)

            if isinstance(normalized, dict):
                clarifications.append({
                    "ambiguous": normalized["ambiguous"],
                    "options": normalized["options"],
                    "quantity": last_qty or 1
                })
                last_qty = None

            elif normalized:
                items.append({
                    "name": normalized,
                    "quantity": last_qty or 1
                })
                last_qty = None

    return {
        "items": items,
        "clarification": clarifications
    }


# -------------------------------
# NER QUALITY SCORER (FLOAT ONLY)
# -------------------------------
def score_ner(entities: Dict[str, Any], intent: str) -> float:
    score = 1.0

    items = entities.get("items", [])
    clarifications = entities.get("clarification", [])

    if not items:
        score -= 0.5

    if intent in {"ADD_ITEM", "REMOVE_ITEM"}:
        for it in items:
            if it["quantity"] <= 0:
                score -= 0.2

    if clarifications:
        score -= min(0.3, 0.1 * len(clarifications))

    return round(max(0.0, min(score, 1.0)), 2)
