# ner/ner_service.py

import spacy
import os
from ner.postprocess import load_menu_items

BASE_DIR = os.path.dirname(__file__)

# -------------------------------
# Load spaCy NER model ONCE
# -------------------------------
NER_MODEL_PATH = os.path.join(BASE_DIR, "models", "ner")
nlp_ner = spacy.load(NER_MODEL_PATH)

# -------------------------------
# Load menu items ONCE
# -------------------------------
MENU_PATH = os.path.join(BASE_DIR, "..", "data", "menu.json")
menu_items = load_menu_items(MENU_PATH)

print("✅ NER model and menu loaded")
