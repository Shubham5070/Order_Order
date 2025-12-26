from services.redis_store import get_json, set_json
from services.menu_service import get_menu

MAX_QTY = 10

def normalize_item_name(name: str) -> str:
    name = name.lower().strip()
    for prefix in ("add ", "remove ", "please ", "can you "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def apply_llm_cart_decision(session_id: str, llm_out: dict):
    cart = get_json(f"cart:{session_id}") or []
    menu = get_menu()

    menu_by_name = {
        item["name"].lower(): item
        for item in menu
    }

    action = llm_out.get("action")
    items = llm_out.get("items", [])

    for i in items:
        raw_name = i.get("name", "")
        name = normalize_item_name(raw_name)
        qty = max(1, min(i.get("quantity", 1), MAX_QTY))

        if name not in menu_by_name:
            continue

        menu_item = menu_by_name[name]

        if action == "ADD_ITEM":
            for c in cart:
                if c["item_id"] == menu_item["id"]:
                    c["quantity"] += qty
                    break
            else:
                cart.append({
                    "item_id": menu_item["id"],
                    "name": menu_item["name"],
                    "price": menu_item["price"],
                    "quantity": qty,
                })

        elif action == "REMOVE_ITEM":
            for idx, c in enumerate(cart):
                if c["item_id"] == menu_item["id"]:
                    c["quantity"] -= qty
                    if c["quantity"] <= 0:
                        cart.pop(idx)
                    break

    set_json(f"cart:{session_id}", cart, ttl=3600)
    return cart
