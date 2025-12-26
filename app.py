from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Any
import uuid

from services.menu_service import get_menu, get_item_by_id
from classifier.intent_minilm import predict_intent
from ner.ner_service import nlp_ner, menu_items
from ner.postprocess import postprocess_ner, score_ner
from llm.ollama_router import run_llm_response
from services.cart_llm_executer import apply_llm_cart_decision

# 🔴 NEW: Redis helpers
from services.redis_store import set_json, get_json, delete
from services.cart_llm_executer import apply_llm_cart_decision


app = FastAPI(title="Restaurant POS Main App")

NER_ACTIONS = {"ADD_ITEM", "REMOVE_ITEM"}

# -------------------------------
# Schemas
# -------------------------------
class StartSessionRequest(BaseModel):
    table_id: str


class AddItemRequest(BaseModel):
    session_id: str
    item_id: str
    quantity: int


class RemoveItemRequest(BaseModel):
    session_id: str
    item_id: str


class ChatRequest(BaseModel):
    message: str


class AgentChatRequest(BaseModel):
    session_id: str
    message: str


# -----------------------
# Serve UI
# -----------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# -------------------------------
# Session APIs
# -------------------------------
@app.post("/session/start")
def start_session(req: StartSessionRequest):
    session_id = str(uuid.uuid4())

    set_json(
        f"session:{session_id}",
        {"table_id": req.table_id, "status": "ORDERING"},
        ttl=3600,
    )

    set_json(f"cart:{session_id}", [], ttl=3600)

    return {
        "session_id": session_id,
        "status": "ORDERING",
    }


# -------------------------------
# Menu API
# -------------------------------
@app.get("/menu")
def get_menu_api():
    return get_menu()


# -------------------------------
# Cart APIs
# -------------------------------
@app.post("/cart/add")
def add_to_cart(req: AddItemRequest):
    session = get_json(f"session:{req.session_id}")
    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")

    if session["status"] != "ORDERING":
        raise HTTPException(status_code=400, detail="Cart already confirmed")

    item = get_item_by_id(req.item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    cart = get_json(f"cart:{req.session_id}") or []

    for cart_item in cart:
        if cart_item["item_id"] == item["id"]:
            cart_item["quantity"] += req.quantity
            break
    else:
        cart.append({
            "item_id": item["id"],
            "name": item["name"],
            "price": item["price"],
            "quantity": req.quantity,
        })

    set_json(f"cart:{req.session_id}", cart, ttl=3600)

    return {"message": "Item added to cart", "items": cart}


@app.post("/cart/remove")
def remove_from_cart(req: RemoveItemRequest):
    session = get_json(f"session:{req.session_id}")
    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")

    if session["status"] != "ORDERING":
        raise HTTPException(status_code=400, detail="Cart already confirmed")

    cart = get_json(f"cart:{req.session_id}") or []

    for i, cart_item in enumerate(cart):
        if cart_item["item_id"] == req.item_id:
            cart_item["quantity"] -= 1
            if cart_item["quantity"] <= 0:
                cart.pop(i)

            set_json(f"cart:{req.session_id}", cart, ttl=3600)
            return {"message": "Item removed", "items": cart}

    raise HTTPException(status_code=404, detail="Item not in cart")


@app.get("/cart/{session_id}")
def view_cart(session_id: str):
    cart = get_json(f"cart:{session_id}")
    if cart is None:
        raise HTTPException(status_code=404, detail="Invalid session")
    return {"items": cart}


@app.post("/cart/confirm")
def confirm_cart(session_id: str):
    session = get_json(f"session:{session_id}")
    cart = get_json(f"cart:{session_id}")

    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")

    if not cart:
        raise HTTPException(status_code=400, detail="Cart is empty")

    session["status"] = "CONFIRMED"
    set_json(f"session:{session_id}", session, ttl=3600)

    return {"message": "Cart confirmed", "items": cart}


# -------------------------------
# Order API
# -------------------------------
@app.post("/order/place")
def place_order(session_id: str):
    session = get_json(f"session:{session_id}")
    cart = get_json(f"cart:{session_id}")

    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")

    if session["status"] != "CONFIRMED":
        raise HTTPException(status_code=400, detail="Confirm cart first")

    order_id = str(uuid.uuid4())

    total = sum(item["price"] * item["quantity"] for item in cart)

    set_json(
        f"order:{order_id}",
        {
            "order_id": order_id,
            "session_id": session_id,
            "table_id": session["table_id"],
            "items": cart,
            "total": total,
            "status": "PLACED",
        },
        ttl=86400,
    )

    session["status"] = "PLACED"
    set_json(f"session:{session_id}", session, ttl=3600)

    return {"order_id": order_id, "session_id": session_id}


@app.get("/order/status/{order_id}")
def order_status(order_id: str):
    order = get_json(f"order:{order_id}")
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

@app.post("/agent/chat")
def agent_chat(req: AgentChatRequest):
    text = req.message.strip()
    session_id = req.session_id

    session = get_json(f"session:{session_id}")
    if not session:
        raise HTTPException(400, "Invalid session")

    # -----------------------------
    # 1️⃣ Intent detection (signal only)
    # -----------------------------
    intent_out = predict_intent(text)
    intent = intent_out["intent"]
    intent_conf = intent_out["confidence"]

    # -----------------------------
    # 2️⃣ NER extraction (signal only)
    # -----------------------------
    from ner.extract_items import extract_items
    ner_result = extract_items(text, menu_items)

    if ner_result.get("food_items"):
        ner_conf = 1.0
    elif ner_result.get("clarification"):
        ner_conf = 0.6
    else:
        ner_conf = 0.3

    # -----------------------------
    # 3️⃣ Decide FLOW (simple & safe)
    # -----------------------------
    if ner_result.get("food_items") and not ner_result.get("clarification"):
        flow = "EXECUTE"
    elif ner_result.get("clarification"):
        flow = "CLARIFICATION"
    else:
        flow = "EXECUTE"

    # -----------------------------
    # 4️⃣ ALWAYS call LLM (NO early return)
    # -----------------------------
    llm_out = run_llm_response(
        user_text=text,
        intent=intent,
        intent_conf=intent_conf,
        ner_result=ner_result,
        ner_conf=ner_conf,
        flow=flow,
        menu_items=menu_items,
    )

    # -----------------------------
    # 5️⃣ Backend cart mutation (ONLY here)
    # -----------------------------
    cart = None
    if (
        intent in ("ADD_ITEM", "REMOVE_ITEM")
        and llm_out.get("action") in ("ADD_ITEM", "REMOVE_ITEM")
        and session["status"] == "ORDERING"
    ):
        cart = apply_llm_cart_decision(session_id, llm_out)

    # -----------------------------
    # 6️⃣ Build response LAST
    # -----------------------------
    response = {
        "intent": intent,
        "intent_confidence": intent_conf,
        "ner": ner_result,
        "ner_confidence": ner_conf,
        "llm": llm_out,
    }

    if cart is not None:
        response["cart"] = cart

    return response


@app.get("/order_status.html", response_class=HTMLResponse)
def order_status_page():
    with open("static/order_status.html", "r", encoding="utf-8") as f:
        return f.read()
