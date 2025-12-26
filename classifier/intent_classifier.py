import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("classifier/models/intent_clf.joblib")

_intent_model = None


def load_intent_model():
    global _intent_model
    if _intent_model is None:
        _intent_model = joblib.load(MODEL_PATH)
    return _intent_model


def predict_intent(text: str) -> dict:
    """
    Returns:
    {
      intent: str,
      confidence: float
    }
    """
    model = load_intent_model()

    probs = model.predict_proba([text])[0]
    idx = int(np.argmax(probs))

    intent = model.classes_[idx]
    confidence = float(probs[idx])

    return {
        "intent": intent,
        "confidence": round(confidence, 3)
    }
