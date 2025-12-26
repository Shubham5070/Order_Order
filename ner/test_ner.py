import spacy
import os

from postprocess import load_menu_items, postprocess_ner, score_ner

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# -------------------------------
# Load spaCy NER model
# -------------------------------
NER_MODEL_PATH = os.path.join(BASE_DIR, "models", "ner")
nlp = spacy.load(NER_MODEL_PATH)

# -------------------------------
# Load menu items
# -------------------------------
MENU_PATH = os.path.join(PROJECT_ROOT, "data", "menu.json")
menu_items = load_menu_items(MENU_PATH)

print("✅ Loaded NER model")
print("✅ Loaded menu items:", len(menu_items))


# -------------------------------
# Test cases
# -------------------------------
tests = [
    ("order coffee", "ADD_ITEM"),
    ("add two paneer tikka pizza", "ADD_ITEM"),
    ("remove one veg burger", "REMOVE_ITEM"),
    ("remove paneer", "REMOVE_ITEM"),     # should trigger clarification
    ("add two dosa", "ADD_ITEM"),          # not in menu → expected empty / ambiguity
]

# -------------------------------
# Run tests
# -------------------------------
for text, intent in tests:
    doc = nlp(text)
    entities = postprocess_ner(doc, menu_items)
    ner_score = score_ner(entities, intent)

    print("\n----------------------------------")
    print("Input:", text)
    print("Intent:", intent)
    print("NER payload:")
    print(entities)
    print("NER score:", ner_score)
