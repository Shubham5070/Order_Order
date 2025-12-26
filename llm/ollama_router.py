from llm.ollama_client import run_ollama
from llm.ollama_parser import extract_json
from llm.ollama_prompts import (
    llm_execute_prompt,
    llm_clarification_prompt,
    llm_suggestion_prompt,
    llm_generic_prompt,
)

def run_llm_response(
    user_text: str,
    intent: str,
    intent_conf: float,
    ner_result: dict,
    ner_conf: float,
    flow: str,
    menu_items: list,
):
    """
    ALWAYS returns a dict with keys:
    - action
    - items
    - message
    """

    # -----------------------------
    # 1. Choose prompt (LLM SPEECH MODE)
    # -----------------------------
    if flow == "EXECUTE":
        prompt = llm_execute_prompt
    elif flow == "CLARIFICATION":
        prompt = llm_clarification_prompt
    elif intent == "SUGGEST_FOOD":
        prompt = llm_suggestion_prompt
    else:
        prompt = llm_generic_prompt

    # -----------------------------
    # 2. Build LLM input (FULL CONTEXT)
    # -----------------------------
    llm_input = f"""
{prompt}

User message:
"{user_text}"

Detected intent:
{intent} (confidence: {intent_conf})

NER result:
{ner_result}

NER confidence:
{ner_conf}

Menu items:
{menu_items}

IMPORTANT RULES:
- You MUST return valid JSON.
- JSON format:

{{
  "action": "ADD_ITEM | REMOVE_ITEM | NONE",
  "items": [{{"name": "<menu item>", "quantity": <int>}}],
  "message": "<short user-facing message>"
}}

- If unsure, set action = "NONE".
- NEVER ask questions outside JSON.
"""

    # -----------------------------
    # 3. Call Ollama (ALWAYS)
    # -----------------------------
    raw_output = run_ollama(llm_input)

    # -----------------------------
    # 4. Parse JSON SAFELY
    # -----------------------------
    try:
        parsed = extract_json(raw_output)

        return {
            "action": parsed.get("action", "NONE"),
            "items": parsed.get("items", []),
            "message": parsed.get(
                "message",
                "Okay, let me know how I can help."
            ),
        }

    except Exception as e:
        # -----------------------------
        # 5. HARD FAILSAFE (NEVER BREAK FLOW)
        # -----------------------------
        return {
            "action": "NONE",
            "items": [],
            "message": "Can you please clarify?",
        }
