from pydantic import BaseModel
from typing import List, Literal

class LLMItem(BaseModel):
    name: str
    quantity: int

class CartDecision(BaseModel):
    action: Literal["ADD_ITEM", "REMOVE_ITEM", "NONE"]
    items: List[LLMItem]
    message: str
