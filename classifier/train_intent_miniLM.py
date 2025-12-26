import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# -----------------------
# Paths
# -----------------------
DATA_PATH = Path("classifier/data/intent_data.csv")
MODEL_PATH = Path("classifier/models/intent_minilm.joblib")

# -----------------------
# Load data (single column)
# -----------------------
df_raw = pd.read_csv(
    DATA_PATH,
    header=None,
    names=["raw"]
)

# Remove accidental header rows
df_raw = df_raw[~df_raw["raw"].str.lower().isin(["text,label", "text,intent"])]

# Split on LAST comma
df = df_raw["raw"].str.rsplit(",", n=1, expand=True)
df.columns = ["text", "label"]

# Clean
df["text"] = df["text"].astype(str).str.lower().str.strip()
df["label"] = df["label"].astype(str).str.upper().str.strip()

# Drop bad rows
df = df.dropna().drop_duplicates()

print("Samples:", len(df))
print(df["label"].value_counts())

texts = df["text"].tolist()
labels = df["label"].tolist()

# -----------------------
# Load embedding model
# -----------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Encode text
X = embedder.encode(texts, show_progress_bar=True)

# -----------------------
# Train classifier
# -----------------------
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
clf.fit(X, labels)

# -----------------------
# Save model
# -----------------------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "embedder": embedder,
        "classifier": clf
    },
    MODEL_PATH
)

print(f"✅ MiniLM intent model saved to {MODEL_PATH}")
