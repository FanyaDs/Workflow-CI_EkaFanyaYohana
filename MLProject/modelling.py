import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ======================
# 1. Load Dataset
# ======================
train_path = "/content/train_preprocess_ori.tsv"
valid_path = "/content/valid_preprocess.tsv"
test_path  = "/content/test_preprocess_masked_label.tsv"

df_train = pd.read_csv(train_path, sep="\t")
df_valid = pd.read_csv(valid_path, sep="\t")
df_test  = pd.read_csv(test_path, sep="\t")

# Gabung train + valid untuk pelatihan final
df_full = pd.concat([df_train, df_valid], axis=0)

# ======================
# 2. Split Data
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    df_full["text"], df_full["sentiment"], test_size=0.2, random_state=42
)

# ======================
# 3. Pipeline Model
# ======================
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=300))
])

# ======================
# 4. Train
# ======================
pipeline.fit(X_train, y_train)

# ======================
# 5. Evaluate
# ======================
preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)

print("📊 Validation Accuracy:", acc)
print(classification_report(y_test, preds))

