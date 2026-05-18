"""Train spam classifier on emails_large.csv (~85k balanced)."""
import re, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA   = "/home/uzair/Desktop/Spam-Filter-AI/data/emails_large.csv"
MODEL  = "/home/uzair/Desktop/Spam-Filter-AI/spam_detector_model.pkl"
VECT   = "/home/uzair/Desktop/Spam-Filter-AI/tfidf_vectorizer.pkl"


def clean(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("Loading...")
df = pd.read_csv(DATA)
print(f"  rows: {len(df)}")
print(df["label"].value_counts())

print("Cleaning...")
df["text"] = df["text"].astype(str).map(clean)
df = df[df["text"].str.len() > 10].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

print("Vectorizing...")
vect = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    stop_words="english",
)
Xtr = vect.fit_transform(X_train)
Xte = vect.transform(X_test)
print(f"  features: {Xtr.shape[1]}")

print("Training LogisticRegression...")
model = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
model.fit(Xtr, y_train)

print("\n=== Evaluation ===")
pred = model.predict(Xte)
print(classification_report(y_test, pred, digits=4))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, pred, labels=["ham", "spam"]))

joblib.dump(model, MODEL)
joblib.dump(vect,  VECT)
print(f"\nSaved -> {MODEL}\nSaved -> {VECT}")
