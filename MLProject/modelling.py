import os
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.multiclass import type_of_target

def resolve_base_dir():
    candidates = []
    try:
        here = Path(__file__).parent
        candidates += [here / "namadataset_preprocessing", here.parent / "namadataset_preprocessing"]
    except NameError:
        cwd = Path.cwd()
        candidates += [cwd / "MLProject" / "namadataset_preprocessing", cwd / "namadataset_preprocessing"]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Folder 'namadataset_preprocessing' tidak ditemukan di lokasi umum.")

def resolve_file(base: Path, preferred_names):
    for name in preferred_names:
        f = base / name
        if f.exists():
            return f
    raise FileNotFoundError(f"Tidak menemukan salah satu file: {preferred_names} di {base}")

base = resolve_base_dir()
train_fp = resolve_file(base, ["train.tsv", "train_preprocess_ori.tsv"])
valid_fp = resolve_file(base, ["valid.tsv", "valid_preprocess.tsv"])
test_fp  = resolve_file(base, ["test.tsv",  "test_preprocess_masked_label.tsv"])

print(f"✅ Dataset folder: {base}")
print(f"  - train: {train_fp.name}")
print(f"  - valid: {valid_fp.name}")
print(f"  - test : {test_fp.name}")

df_train = pd.read_csv(train_fp, sep="\t")
df_valid = pd.read_csv(valid_fp, sep="\t")
df_test  = pd.read_csv(test_fp,  sep="\t")

print("Train shape:", df_train.shape, "| Valid shape:", df_valid.shape, "| Test shape:", df_test.shape)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

pipeline.fit(df_train["text"], df_train["sentiment"])

y_val = pipeline.predict(df_valid["text"])
val_acc = accuracy_score(df_valid["sentiment"], y_val)
print("\n📊 Validation Accuracy:", val_acc)
print(classification_report(df_valid["sentiment"], y_val))

has_valid_test_labels = (
    "sentiment" in df_test.columns
    and df_test["sentiment"].notna().any()
    and type_of_target(df_test["sentiment"]) == "multiclass"
)

if has_valid_test_labels:
    y_te = pipeline.predict(df_test["text"])
    te_acc = accuracy_score(df_test["sentiment"], y_te)
    print("\n📊 Test Accuracy:", te_acc)
    print(classification_report(df_test["sentiment"], y_te))
else:
    print("\nℹ️ Test labels are masked/unavailable. Skipping test evaluation.")
