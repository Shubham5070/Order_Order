import json
import ollama
from llm.schemas import CartDecision

SYSTEM_PROMPT = """
You are a restaurant ordering assistant.

Rules:
- You can only decide cart changes.
- Never confirm order.
- Never place order.
- Never talk about payment.

Return ONLY valid JSON:

{
  "action": "ADD_ITEM | REMOVE_ITEM | NONE",
  "items": [{"name": "<exact menu item>", "quantity": <int>}],
  "message": "<short reply>"
}
"""

def decide_cart_action(user_text: str, menu_items: list[str]) -> CartDecision:
    prompt = f"""
Menu:
{menu_items}

User said:
"{user_text}"
"""

    response = ollama.generate(
        model="llama3.2",
        prompt=SYSTEM_PROMPT + prompt,
        options={"temperature": 0.2}
    )

    try:
        return CartDecision(**json.loads(response["response"]))
    except Exception:
        # Hard safety fallback
        return CartDecision(
            action="NONE",
            items=[],
            message="Sorry, I didn’t quite get that. Could you rephrase?"
        )
