import re
from rapidfuzz import process, fuzz

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

GENERIC_HEADS = {"paneer", "burger", "sandwich", "pizza", "coffee"}

def extract_quantity(text: str) -> int | None:
    tokens = re.findall(r"\b\w+\b", text.lower())
    for t in tokens:
        if t.isdigit():
            return int(t)
        if t in WORD_TO_NUM:
            return WORD_TO_NUM[t]
    return None


def extract_items(text: str, menu_items: list[str]):
    text = text.lower()
    quantity = extract_quantity(text) or 1

    found_items = []
    ambiguities = []

    tokens = re.findall(r"\b\w+\b", text)

    # --- Exact & plural match ---
    for item in menu_items:
        if item in text:
            found_items.append(item)

    # --- Token-wise fuzzy match (fallback) ---
    if not found_items:
        for t in tokens:
            match, score, _ = process.extractOne(
                t, menu_items, scorer=fuzz.partial_ratio
            ) or (None, 0, None)

            if score >= 85:
                found_items.append(match)

    # --- Generic head ambiguity ---
    for t in tokens:
        if t in GENERIC_HEADS:
            options = [m for m in menu_items if t in m]
            if options:
                ambiguities.append({
                    "ambiguous": t,
                    "options": options
                })

    # --- Final result ---
    if ambiguities:
        return {
            "quantity": quantity,
            "food_items": [],
            "clarification": ambiguities
        }

    return {
        "quantity": quantity,
        "food_items": list(set(found_items)),
        "clarification": []
    }
