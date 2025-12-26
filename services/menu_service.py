import json
from pathlib import Path
from typing import List, Dict

MENU_PATH = Path("data/menu.json")

_menu_cache: List[Dict] = []


def load_menu() -> List[Dict]:
    global _menu_cache

    if _menu_cache:
        return _menu_cache

    with open(MENU_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    _menu_cache = data["items"]
    return _menu_cache


def get_menu() -> List[Dict]:
    return load_menu()


def get_item_by_id(item_id: str) -> Dict | None:
    return next(
        (item for item in load_menu() if item["id"] == item_id),
        None
    )
