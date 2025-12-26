import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import json
import random
import os

BASE_DIR = os.path.dirname(__file__)

# -------------------------------
# Load training data
# -------------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "train_ner_data.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    TRAIN_DATA = json.load(f)

# -------------------------------
# Create spaCy pipeline
# -------------------------------
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner", last=True)

ner.add_label("FOOD_ITEM")
ner.add_label("QUANTITY")

nlp.initialize()

# -------------------------------
# Convert to spaCy Examples
# -------------------------------
examples = []

for row in TRAIN_DATA:
    doc = nlp.make_doc(row["text"])
    example = Example.from_dict(doc, {"entities": row["entities"]})
    examples.append(example)

print(f"✅ Loaded {len(examples)} training examples")

# -------------------------------
# Training loop
# -------------------------------
EPOCHS = 12
BATCH_SIZE = 16
best_loss = float("inf")

MODEL_DIR = os.path.join(BASE_DIR, "models", "ner")
os.makedirs(MODEL_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    random.shuffle(examples)
    losses = {}

    for batch in minibatch(examples, size=BATCH_SIZE):
        nlp.update(batch, drop=0.2, losses=losses)

    loss = losses.get("ner", 0.0)
    print(f"Epoch {epoch + 1} | Loss: {loss:.4f}")

    if loss < best_loss:
        best_loss = loss
        nlp.to_disk(MODEL_DIR)
        print("✅ Best model saved")

print("🎉 NER training completed")
