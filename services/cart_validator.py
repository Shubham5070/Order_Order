MAX_QTY = 10

def validate_items(items, menu_set):
    valid = []
    for item in items:
        if (
            item.name in menu_set
            and 1 <= item.quantity <= MAX_QTY
        ):
            valid.append(item)
    return valid
