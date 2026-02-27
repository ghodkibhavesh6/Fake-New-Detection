# train_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("news_data.csv", encoding="latin1")

# =========================
# CLEAN TEXT
# =========================
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# =========================
# FIX LABEL COLUMN
# =========================
df["label"] = df["label"].astype(str).str.upper()

label_map = {
    "REAL": 1,
    "FAKE": 0,
    "1": 1,
    "0": 0
}

df["label"] = df["label"].map(label_map)

# remove invalid rows
df = df.dropna(subset=["label"])

# =========================
# CREATE CONTENT
# =========================
df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = MultinomialNB()
model.fit(X_train, y_train)

# =========================
# ACCURACY
# =========================
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# =========================
# SAVE FILES
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model saved successfully")