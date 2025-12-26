"""
Optimized prompt templates.
Same structure, same use-cases, fewer tokens.
"""
import json

# -------------------------
# Compact helpers
# -------------------------

def ner_signal(ner: dict | None) -> str:
    """
    Compress NER payload into a single short line.
    """
    if not ner:
        return "none"

    items = ner.get("food_items") or ner.get("items") or []
    qty = ner.get("quantity", 1)
    clar = ner.get("clarification")

    if clar:
        amb = clar[0].get("ambiguous", "unknown")
        return f"ambiguous:{amb}"

    if items:
        if isinstance(items[0], dict):
            names = [i["name"] for i in items]
        else:
            names = items
        return f"items:{','.join(names)} x{qty}"

    return "none"


def menu_signal(menu: list, limit: int = 6) -> str:
    """
    Only expose a few menu items.
    """
    if not menu:
        return ""
    return ", ".join(menu[:limit])


# -------------------------
# Shared rule (compressed)
# -------------------------

BASE_RULES = (
    "Rules:no intent change,no execute,menu-only,JSON-only,short."
)

# -------------------------
# EXECUTE prompt
# -------------------------

def llm_execute_prompt(user_text, intent, intent_conf, ner, ner_conf):
    return f"""
Mode:EXECUTE
Intent:{intent} ({intent_conf:.2f})
NER:{ner_signal(ner)}

User:"{user_text}"

Task:Explain what will happen.

{BASE_RULES}

Return:{{"message":"...","role":"EXECUTE"}}
"""

# -------------------------
# CLARIFICATION prompt
# -------------------------

def llm_clarification_prompt(user_text, intent, intent_conf, ner, ner_conf, menu):
    return f"""
Mode:CLARIFICATION
Intent:{intent} ({intent_conf:.2f})
NER:{ner_signal(ner)}
Menu:{menu_signal(menu)}

User:"{user_text}"

Task:Ask ONE clarification question.

{BASE_RULES}

Return:{{"message":"...","role":"CLARIFICATION"}}
"""

# -------------------------
# SUGGESTION prompt
# -------------------------

def llm_suggestion_prompt(user_text, intent, intent_conf, menu):
    return f"""
Mode:SUGGEST
Intent:{intent} ({intent_conf:.2f})
Menu:{menu_signal(menu)}

User:"{user_text}"

Task:Suggest 2–3 items.

{BASE_RULES}

Return:{{"message":"...","role":"SUGGEST"}}
"""

# -------------------------
# GENERIC prompt
# -------------------------

def llm_generic_prompt(user_text, intent, intent_conf):
    return f"""
Mode:GENERIC
Intent:{intent} ({intent_conf:.2f})

User:"{user_text}"

Task:Respond politely.

{BASE_RULES}

Return:{{"message":"...","role":"GENERIC"}}
"""

def llm_cart_prompt(user_text, menu):
    return f"""
You are a restaurant ordering assistant.

You MAY:
- Add items to cart
- Remove items from cart

You MUST NOT:
- Confirm cart
- Place order
- Invent menu items

Menu (exact names only):
{json.dumps([m["name"] for m in menu])}

User message:
"{user_text}"

Return ONLY JSON in this format:
{{
  "action": "ADD_ITEM | REMOVE_ITEM | NONE",
  "items": [
    {{ "name": "<menu item>", "quantity": 1 }}
  ],
  "message": "<short reply to user>"
}}

Rules:
- If unsure, set action = "NONE"
- Never explain reasoning
- Never include markdown
"""
