"""Build a balanced ~86k spam/ham dataset.

Sources:
  - 43k spam from Kaggle combined_data.csv (label=1)
  - 23k ham  from Enron sample (enron_ham_50k.csv)
  - 20k ham  from Kaggle combined_data.csv (label=0)
"""
import csv, random, sys

csv.field_size_limit(sys.maxsize)

ENRON  = "/home/uzair/Desktop/Spam-Filter-AI/data/enron_ham_50k.csv"
KAGGLE = "/home/uzair/Desktop/Spam-Filter-AI/data/combined_data.csv"
OUT    = "/home/uzair/Desktop/Spam-Filter-AI/data/emails_large.csv"

ENRON_HAM   = 23000
KAGGLE_HAM  = 20000
random.seed(42)


def clean(t):
    t = (t or "").strip()
    return t if len(t) >= 20 else None


# 1. Load Kaggle, split spam/ham
print("Loading Kaggle...")
kaggle_spam, kaggle_ham = [], []
with open(KAGGLE, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        t = clean(r["text"])
        if not t:
            continue
        (kaggle_spam if r["label"] == "1" else kaggle_ham).append(t)
print(f"  spam: {len(kaggle_spam)}, ham: {len(kaggle_ham)}")

# 2. Load Enron ham
print("Loading Enron...")
enron_ham = []
with open(ENRON, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        t = clean(r["text"])
        if t:
            enron_ham.append(t)
print(f"  ham: {len(enron_ham)}")

# 3. Sample
spam_sample       = kaggle_spam                                                # use all
enron_sample      = random.sample(enron_ham,  min(ENRON_HAM,  len(enron_ham)))
kaggle_ham_sample = random.sample(kaggle_ham, min(KAGGLE_HAM, len(kaggle_ham)))

rows = (
    [(t, "spam") for t in spam_sample]
    + [(t, "ham") for t in enron_sample]
    + [(t, "ham") for t in kaggle_ham_sample]
)

# 4. Deduplicate (key on first 500 chars)
print("Deduplicating...")
seen, deduped = set(), []
for text, label in rows:
    key = text[:500]
    if key in seen:
        continue
    seen.add(key)
    deduped.append((text, label))

random.shuffle(deduped)

# 5. Write
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    w.writerows(deduped)

ham  = sum(1 for _, l in deduped if l == "ham")
spam = sum(1 for _, l in deduped if l == "spam")
print(f"Done -> {OUT}")
print(f"  total: {len(deduped)}  ham: {ham}  spam: {spam}")
