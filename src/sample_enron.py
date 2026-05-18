"""Sample 50k ham emails from the Kaggle Enron CSV (file,message)."""
import csv, random, sys

csv.field_size_limit(sys.maxsize)

INPUT  = "/home/uzair/Desktop/Spam-Filter-AI/data/emails.csv"
OUTPUT = "/home/uzair/Desktop/Spam-Filter-AI/data/enron_ham_50k.csv"
SAMPLE_SIZE = 50000
random.seed(42)


def extract(raw):
    """Split headers from body the simple way: first blank line."""
    if not raw:
        return None
    parts = raw.split("\n\n", 1)
    if len(parts) < 2:
        return None
    headers, body = parts

    subject = ""
    for line in headers.splitlines():
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            break

    text = (subject + "\n" + body).strip()
    return text if len(text) > 20 else None


print("Counting rows...")
with open(INPUT, encoding="utf-8") as f:
    total = sum(1 for _ in csv.DictReader(f))
print(f"Total rows: {total}")

picked = set(random.sample(range(total), min(SAMPLE_SIZE, total)))

print("Sampling + parsing...")
kept = 0
with open(INPUT, encoding="utf-8") as fin, \
     open(OUTPUT, "w", newline="", encoding="utf-8") as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["text", "label"])
    for i, row in enumerate(reader):
        if i not in picked:
            continue
        text = extract(row.get("message", ""))
        if text:
            writer.writerow([text, "ham"])
            kept += 1

print(f"Wrote {kept} ham emails -> {OUTPUT}")
