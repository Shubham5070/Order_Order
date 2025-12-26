import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("classifier/models/intent_minilm.joblib")

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_intent(text: str):
    model = load_model()
    embedder = model["embedder"]
    clf = model["classifier"]

    emb = embedder.encode([text.lower()])[0]
    probs = clf.predict_proba([emb])[0]

    idx = int(np.argmax(probs))

    return {
        "intent": clf.classes_[idx],
        "confidence": round(float(probs[idx]), 3),
        "alternatives": [
            (clf.classes_[i], round(float(probs[i]), 3))
            for i in range(len(probs))
        ]
    }
