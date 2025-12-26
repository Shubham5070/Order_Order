import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(
    "classifier/data/intent_data.csv",
    header=None,
    names=["raw"]
)

df[["text", "label"]] = df["raw"].str.rsplit(",", n=1, expand=True)

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()

# --------------------------------------------------
# Pipeline
# --------------------------------------------------
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            C=2.0,
            random_state=42
        )
    )
])

# --------------------------------------------------
# Train
# --------------------------------------------------
pipeline.fit(texts, labels)

# --------------------------------------------------
# Save
# --------------------------------------------------
MODEL_PATH = "classifier/models/intent_clf.joblib"
joblib.dump(pipeline, MODEL_PATH, compress=3)

print(f"✅ Intent classifier saved to {MODEL_PATH}")
