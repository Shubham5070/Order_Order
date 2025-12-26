from services.redis_store import get_cart, save_cart

def apply_cart_action(session_id, action, items):
    cart = get_cart(session_id)

    if action == "ADD_ITEM":
        for item in items:
            cart[item.name] = cart.get(item.name, 0) + item.quantity

    elif action == "REMOVE_ITEM":
        for item in items:
            if item.name in cart:
                cart[item.name] -= item.quantity
                if cart[item.name] <= 0:
                    del cart[item.name]

    save_cart(session_id, cart)
    return cart
