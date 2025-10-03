import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset dummy (ganti dengan dataset kamu kalau mau)
data = {
    "text": ["saya suka produk ini", "saya benci layanan ini", "cukup bagus lah"],
    "sentiment": ["positive", "negative", "neutral"]
}
df = pd.DataFrame(data)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)

# 3. Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200))
])

# 4. Train
pipeline.fit(X_train, y_train)

# 5. Evaluate
preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)
